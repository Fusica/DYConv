import time

import torch
import torch.nn as nn


# spatial_attention = torch.randn(1, 1, 1, 1, 3, 3)
# kernel_attention = torch.randn(1, 4, 1, 1, 1, 1)
# x = torch.randn(4, 256, 128, 3, 3)
#
# y = x.unsqueeze(dim=0) * spatial_attention * kernel_attention
#
# y = torch.sum(y, dim=1)

# for i in range(2):
#     print(i)

# print(y.shape)


# t3 = torch.arange(1, 257).reshape(4, 4, 4, 4)
# print(t3)
# # tensor([[[ 1,  2,  3],
# #         [ 4,  5,  6],
# #         [ 7,  8,  9]],
#
# #        [[10, 11, 12],
# #         [13, 14, 15],
# #         [16, 17, 18]],
#
# #        [[19, 20, 21],
# #         [22, 23, 24],
# #         [25, 26, 27]]])
#
#
# print(t3[:, :, ::3, ::3])
# '''索引第二个矩阵，行和列都是每隔两个取一个'''
# # tensor([[10, 12],
# #        [16, 18]])


# import torch
#
# # x = torch.tensor([[1, 2], [3, 4]])
# # y = x.expand_as(torch.zeros(2, 2, 2, 2))
# # print(y)
#
# m = nn.AdaptiveMaxPool2d(7)
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# print(output.shape)


# # Define input tensor
# input = torch.randn(1, 3, 64, 64)
#
# # Define convolutional filter
# filter = torch.randn(8, 3, 3, 3)
#
# # Define offset tensor
# offset = torch.randn(1, 8, 64, 64, 2)
#
# # Perform deformable convolution
# output = torchvision.ops.deform_conv2d(input, filter, offset)
#
# # Print output tensor shape
# print(output.shape)


# input = torch.rand(4, 3, 10, 10)
# kh, kw = 3, 3
# weight = torch.rand(5, 3, kh, kw)
# # offset and mask should have the same spatial size as the output
# # of the convolution. In this case, for an input of 10, stride of 1
# # and kernel size of 3, without padding, the output size is 8
# offset = torch.rand(4, 2 * kh * kw, 8, 8)
# mask = torch.rand(4, kh * kw, 8, 8)
# out = torchvision.ops.deform_conv2d(input, offset, weight, mask=mask)
# print(out.shape)


# x = torch.randn(1, 128, 160, 160)
#
# model = torchvision.ops.DeformConv2d(128, 256, 3, 1, 1)
# y = model(x)
# print(y.shape())


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


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

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

        return out


class HPool(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
        self.ca = CoordAtt(c1, c1)
        self.pool3 = nn.AdaptiveMaxPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.cv = nn.Conv2d(2 * c1, c2, 1)

    def forward(self, x):
        x_1 = self.pool3(self.ca(self.pool1(x)))
        x_2 = self.pool4(self.pool2(x))
        return self.cv(torch.cat((x_1, x_2), dim=1))


if __name__ == '__main__':
    model = CoordAtt(128, 128)

    x = torch.tensor([[[[0.25]]]])
    x1 = torch.randn(1, 128, 160, 160)
    x2 = torch.randn(4, 256, 1, 1)

    start = time.time()
    y = model(x1)
    end = time.time()
    print(y.shape)
    print(end - start)
