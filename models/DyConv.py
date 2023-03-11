import thop
import torch.autograd
from torch.nn.init import calculate_gain

from models.ODConv import *
from models.common import Conv


# from models.experimental import MAPool, MAdaPool


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


class Gating(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Gating, self).__init__()
        self.inp = inp
        self.oup = oup
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.cv2 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(inp, oup, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.inp == self.oup:
            identity = x
        else:
            identity = self.cv2(x)

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

        out = identity * a_w * a_h

        out = self.relu(self.fc(self.pool(x)))

        return out


class Attention_Fusion(nn.Module):
    def __init__(self, c1, c2, temperature):
        super().__init__()
        c_ = c1 + c2
        self.temperature = temperature
        self.fc = nn.Sequential(
            nn.Linear(c_, c_ // 16),
            nn.ReLU(inplace=True),
            nn.Linear(c_ // 16, c_),
            nn.Sigmoid()
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


def build_in_channel_branch(in_channels, se_ratio=0.25):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, mid_channels, 1, bias=False),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, in_channels, 1)
    )


def build_out_channel_branch(in_channels, out_channels, se_ratio=0.25):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, mid_channels, 1),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, out_channels, 1)
    )


class In_Channel_Att(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.in_att = build_in_channel_branch(c1)

    def forward(self, x):
        in_att = torch.sigmoid(self.in_att(x))

        return in_att


class Out_Channel_Att(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.out_att = build_out_channel_branch(c1, c2)

    def forward(self, x):
        out_att = torch.sigmoid(self.out_att(x))

        return out_att


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, temperature=1, kernel_num=4,
                 min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.gating = Gating(in_planes, attention_channel)

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size ** 2, 1, bias=True)
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

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.gating(x)
        return self.func_spatial(x), self.func_kernel(x)


class MMDyConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
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
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, temperature=temperature, kernel_num=kernel_num)
        self.inc_att = In_Channel_Att(in_planes)
        self.outc_att = Out_Channel_Att(in_planes, out_planes)
        self.fuse = Attention_Fusion(in_planes, out_planes, temperature)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        # channel_attention: (1, 128, 1, 1)
        # filter_attention:  (1, 256, 1, 1)
        # spatial_attention: (1, 1, 1, 1, 3, 3)
        # kernel_attention:  (1, 4, 1, 1, 1, 1)
        spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        channel_att = self.inc_att(x)
        filter_att = self.outc_att(x)
        w_1, w_2 = self.fuse(channel_att, filter_att)
        x = w_1 * x
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = w_2 * output
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class MMConv(Conv):
    def __init__(self, c1, c2, k, s=1, p=(0, 0), temperature=1):
        super().__init__(c1, c2, k, s, p)
        self.conv = MMDyConv2d(c1, c2, k, s, p, temperature=temperature)


class SPPCSPC_Dy(SPPCSPC):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), temperature=31):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.temperature = temperature
        # self.cv1 = MMConv(c1, c_, 1, temperature=temperature)
        # self.cv2 = MMConv(c1, c_, 1, temperature=temperature)
        self.cv3 = MMConv(c_, c_, 3, 1, 1, temperature=temperature)
        # self.cv4 = MMConv(c_, c_, 1, temperature=temperature)
        # self.m = nn.ModuleList([MAPool(c_, c_, kernel=x, stride=1, padding=x // 2) for x in k])
        # self.cv5 = MMConv(4 * c_, c_, 1, temperature=temperature)
        self.cv6 = MMConv(c_, c_, 3, 1, 1, temperature=temperature)
        # self.cv7 = MMConv(2 * c_, c2, 1, temperature=temperature)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))


class Dy_ELAN(nn.Module):
    def __init__(self, c1, c2, e=0.5, temperature=31):
        super().__init__()
        c_ = int(c1 * e)
        self.temperature = temperature
        self.cv1 = Conv(c1, c1, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.m1 = MMConv(c_, c_, 3, 1, 1, temperature=temperature)
        self.m2 = MMConv(c_, c_, 3, 1, 1, temperature=temperature)
        self.m3 = MMConv(c_, c_, 3, 1, 1, temperature=temperature)
        self.m4 = nn.Conv2d(c_, c_, 3, 1, 1)
        self.m5 = nn.Conv2d(c_, c_, 3, 1, 2, 2)
        self.m6 = nn.Conv2d(c_, c_, 3, 1, 5, 5)
        self.cv3 = Conv(4 * c_, c_, 1)
        self.cv4 = Conv(3 * c_, c_, 1)
        self.cv5 = Conv(2 * c_, c2, 1)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        x2 = self.m1(x1)
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        x_dy = torch.cat((x1, x2, x3, x4), dim=1)
        x_dy = self.cv3(x_dy)

        x_5 = self.m4(x1)
        x_6 = self.m5(x1)
        x_7 = self.m6(x1)
        x_di = torch.cat((x_5, x_6, x_7), dim=1)
        x_di = self.cv4(x_di)
        x_all = torch.cat((x_dy, x_di), dim=1)
        return self.cv5(x_all)


if __name__ == '__main__':
    in1 = torch.randn(4, 128, 160, 160)
    in2 = torch.randn(4, 128, 160, 160)
    model = Dy_ELAN(128, 128)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
                 with_flops=True, with_modules=True) as prof:
        model(in1)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    flops, params = thop.profile(model, inputs=(in1,))
    print('FLOPs: ' + str(flops) + ', Params: ' + str(params))
