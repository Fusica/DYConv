import torch.autograd

from models.common import Conv
from models.Deformable_Conv import *
from models.ODConv import *
from models.experimental import MAPool, MAdaPool


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


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, temperature=1, kernel_num=4,
                 min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size[0] == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_cv = Deformable_Conv2D(attention_channel, kernel_size[0] * kernel_size[1], 1, 1, 0)
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
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1))
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1))
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_cv(x).view(x.size(0), 1, 1, 1, self.kernel_size[0], self.kernel_size[1])
        spatial_attention = torch.sigmoid(spatial_attention)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        # x = self.bn(x)
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
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, temperature=temperature, kernel_num=kernel_num)
        self.attention_fusion = Attention_Fusion(in_planes, out_planes, temperature=temperature)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size[0], kernel_size[1]),
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
        # channel_attention: (1, 128, 1, 1)
        # filter_attention:  (1, 256, 1, 1)
        # spatial_attention: (1, 1, 1, 1, 3, 3)
        # kernel_attention:  (1, 4, 1, 1, 1, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        w1, w2 = self.attention_fusion(channel_attention, filter_attention)
        x = w1 * x * channel_attention  # channel_attention: (1, 128, 1, 1)
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size[0], self.kernel_size[1]])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=(self.padding[0], self.padding[1]),
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = w2 * output * filter_attention
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
        self.m1 = nn.Sequential(
            MMConv(c_, c_, (3, 3), 1, (1, 1), temperature=temperature),
            MMConv(c_, c1, (3, 3), 1, (1, 1), temperature=temperature)
        )
        self.m2 = nn.Sequential(
            MMConv(c1, c1, (5, 1), 1, (2, 0), temperature=temperature),
            MMConv(c1, c1, (1, 5), 1, (0, 2), temperature=temperature),
        )
        self.m3 = nn.Sequential(
            MMConv(c1, c1, (7, 1), 1, (3, 0), temperature=temperature),
            MMConv(c1, c1, (1, 7), 1, (0, 3), temperature=temperature),
        )
        self.ma1 = MAPool(c1, c2, 9, 1, 4)
        self.ma2 = MAPool(c1, c2, 13, 1, 6)
        self.cv3 = Conv(6 * c1, c2, 1)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m1(self.cv2(x))
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        x5 = self.ma1(x4)
        x6 = self.ma2(x5)
        x_all = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        return self.cv3(x_all)


if __name__ == '__main__':
    x = torch.randn(4, 512, 20, 20)
    model = Dy_ELAN(512, 512)

    start = time.time()
    y = model(x)
    end = time.time()
    print(y.shape)
    print(end - start)

    flops, params = thop.profile(model, inputs=(x,))
    print('FLOPs: ' + str(flops) + ', Params: ' + str(params))
