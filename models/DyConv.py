import thop
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple
from torch.profiler import profile, ProfilerActivity

from models.common import DSigmoid, DReLU
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
        if isinstance(filter_att, float):
            return 1.0, 1.0
        else:
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



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Spatial_attn(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Spatial_attn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = a_w * a_h

        return out


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
        self.spatial_attn = Spatial_attn(in_planes, out_planes)
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
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
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


class InceptionDY(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, c1, c2, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, temperature=31):
        super().__init__()

        self.temperature = temperature
        gc = int(c1 * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = MMDyConv2d(gc, gc, (square_kernel_size, square_kernel_size),
                                    padding=(square_kernel_size // 2, square_kernel_size // 2), temperature=temperature)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = [c1 - 3 * gc, gc, gc, gc]

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 2
            print("temperature changed to {}".format(self.temperature))


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = DReLU(hidden_features)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DYBlock(nn.Module):
    def __init__(self, c1, c2, temperature=31):
        super().__init__()
        self.temperature = temperature
        self.inception = InceptionDY(c1, c2, temperature=temperature)
        self.mlp = ConvMlp(c2, c2, c2, drop=0.)

    def forward(self, x):
        return self.mlp(self.inception(x))

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 2
            print("temperature changed to {}".format(self.temperature))


if __name__ == '__main__':
    in1 = torch.randn(4, 1024, 20, 20)
    in2 = torch.randn(4, 4, 64, 64, 3, 3)
    model = DYBlock(1024, 1024)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
                 with_flops=True, with_modules=True) as prof:
        y = model(in1)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    flops, params = thop.profile(model, inputs=(in1,))
    print('FLOPs: ' + str(flops) + ', Params: ' + str(params))
