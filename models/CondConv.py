import time

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, in_planes, K, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if init_weight:
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

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return self.sigmoid(att)


class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=(0, 0), dilation=1, groups=1, bias=True,
                 num_exp=4,
                 init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_exp = num_exp
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, K=num_exp, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(num_exp, out_planes, in_planes // groups, kernel_size[0], kernel_size[1]),
                                   requires_grad=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.randn(num_exp, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_exp):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.num_exp, -1)  # K,-1
        """
        Combination
        """
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size[0],
                                                              self.kernel_size[1])  # bs*out_p,in_p,k,k

        if self.bias is not None:
            bias = self.bias.view(self.num_exp, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                              padding=(self.padding[0], self.padding[1]),
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                              padding=(self.padding[0], self.padding[1]),
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return self.act(self.bn(output))


if __name__ == '__main__':
    input = torch.randn(1, 128, 160, 160)
    m = CondConv(in_planes=128, out_planes=256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False, num_exp=4)
    start = time.time()
    out = m(input)
    end = time.time()
    print(out.shape)
    print(end - start)
