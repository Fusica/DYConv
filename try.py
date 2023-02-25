import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == '__main__':
    model = Attention_Fusion(128, 256)

    x = torch.tensor([[[[0.25]]]])
    x1 = torch.randn(4, 128, 1, 1)
    x2 = torch.randn(4, 256, 1, 1)

    start = time.time()
    y = model(x1, x2)
    end = time.time()
    print(y[0])
    print(end - start)
