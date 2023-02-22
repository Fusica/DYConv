import torch
import torch.nn as nn
import numpy as np

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


t3 = torch.arange(1, 257).reshape(4, 4, 4, 4)
print(t3)
# tensor([[[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9]],

#        [[10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]],

#        [[19, 20, 21],
#         [22, 23, 24],
#         [25, 26, 27]]])


print(t3[:, :, ::3, ::3])
'''索引第二个矩阵，行和列都是每隔两个取一个'''
# tensor([[10, 12],
#        [16, 18]])