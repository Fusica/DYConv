import thop
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from models.common import Conv, DSigmoid, DReLU
from models.experimental import DSConv


class Attention_Fusion(nn.Module):
    def __init__(self, c1, c2, temperature):
        super().__init__()
        c_ = c1 + c2
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(c_, c_ // 16),
            nn.ReLU(inplace=True),
            nn.Linear(c_ // 16, c_),
            DSigmoid()
        )

    def forward(self, channel_att, filter_att):
        b = channel_att.size(0)
        c_c, f_c = channel_att.size(1), filter_att.size(1)
        att = torch.cat((channel_att, filter_att), dim=1).view(b, -1)
        att = self.fc(att).view(b, -1, 1, 1)
        att = torch.split(att, [c_c, f_c], dim=1)
        score_1 = torch.sum(att[0], dim=1, keepdim=True)
        score_2 = torch.sum(att[1], dim=1, keepdim=True)
        score = torch.cat((score_1, score_2), dim=1)
        score = F.softmax(score / self.temperature, dim=1)
        w_1 = score[:, 0, :, :].view(b, 1, 1, 1)
        w_2 = score[:, 1, :, :].view(b, 1, 1, 1)
        return w_1, w_2


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class Spatial_attn(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Channel_attn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temperature=1.0):
        super(Channel_attn, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = DReLU(attention_channel)
        self.act = DSigmoid()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size[0] * kernel_size[1], 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = self.act(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = self.act(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size[0], self.kernel_size[1])
        spatial_attention = self.act(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class MMDyConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3), stride=1, padding=(0, 0), dilation=1, groups=1,
                 reduction=0.0625, temperature=1, kernel_num=4):
        super(MMDyConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.temperature = temperature
        self.kernel_num = kernel_num
        self.attention = Channel_attn(in_planes, out_planes, kernel_size, groups=groups,
                                      reduction=reduction, temperature=temperature, kernel_num=kernel_num)
        self.fuse = Attention_Fusion(in_planes, out_planes, temperature)
        self.spatial_attn = Spatial_attn(in_planes, out_planes, 4, 4)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size[0], kernel_size[1]),
            requires_grad=True)
        self._initialize_weights()

        if self.kernel_size[0] == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        # spatial_attention: (1, 1, 1, 1, 3, 3)
        # kernel_attention:  (1, 4, 1, 1, 1, 1)
        spatial_attn = self.spatial_attn(x)
        channel_attention, filter_attention, spatial_attention, kernel_attention, = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        w_1, w_2 = self.fuse(channel_attention, filter_attention)
        x = w_1 * x
        x = x.reshape(1, -1, height, width)

        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size[0], self.kernel_size[1]])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                          padding=(self.padding[0], self.padding[1]),
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = spatial_attn * (w_2 * output)
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride,
                          padding=(self.padding[0], self.padding[1]),
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class MMConv(nn.Module):
    def __init__(self, c1, c2, k=(3, 3), s=1, p=(0, 0), temperature=1):
        super().__init__()
        self.conv = MMDyConv2d(c1, c2, k, s, p, temperature=temperature)
        self.bn = nn.BatchNorm2d(c2)
        self.act = DReLU(c2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Dy_ELAN(nn.Module):
    def __init__(self, c1, c2, e=0.5, temperature=31):
        super().__init__()
        c_ = int(c1 * e)
        self.temperature = temperature
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            DReLU(c1)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            DReLU(c_)
        )
        self.m1 = nn.Sequential(
            MMConv(c_, c_, (3, 3), 1, (1, 1), temperature=temperature)
        )
        self.m2 = nn.Sequential(
            MMConv(c_, c_, (3, 3), 1, (1, 1), temperature=temperature)
        )
        self.m3 = nn.Sequential(
            MMConv(c_, c_, (3, 3), 1, (1, 1), temperature=temperature)
        )
        self.m4 = DSConv(c_, c_, 13, 1, 6)
        self.cv3 = nn.Sequential(
            nn.Conv2d(4 * c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            DReLU(c_)
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(2 * c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            DReLU(c2)
        )

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 0.2

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        x2 = self.m1(x1)
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        x_dy = torch.cat((x1, x2, x3, x4), dim=1)
        x_dy = self.cv3(x_dy)

        x_di = self.m4(x1)

        x_all = torch.cat((x_dy, x_di), dim=1)
        return self.cv4(x_all)


if __name__ == '__main__':
    in1 = torch.randn(4, 1024, 20, 20)
    in2 = torch.randn(4, 4, 64, 64, 3, 3)
    model = Dy_ELAN(1024, 1024)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
                 with_flops=True, with_modules=True) as prof:
        y = model(in1)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    flops, params = thop.profile(model, inputs=(in1,))
    print('FLOPs: ' + str(flops) + ', Params: ' + str(params))
