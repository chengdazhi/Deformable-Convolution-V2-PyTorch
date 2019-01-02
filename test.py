import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import DeformConv

num_deformable_groups = 2

N, inC, inH, inW = 2, 6, 512, 512
outC, outH, outW = 4, 512, 512
kH, kW = 3, 3

conv = nn.Conv2d(
    inC,
    num_deformable_groups * 2 * kH * kW,
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=(1, 1),
    bias=False).cuda()

conv_offset2d = DeformConv(
    inC,
    outC, (kH, kW),
    stride=1,
    padding=1,
    num_deformable_groups=num_deformable_groups).cuda()

inputs = Variable(torch.randn(N, inC, inH, inW).cuda(), requires_grad=True)
offset = conv(inputs)
#offset = Variable(torch.randn(N, num_deformable_groups * 2 * kH * kW, inH, inW).cuda(), requires_grad=True)
output = conv_offset2d(inputs, offset)
output.backward(output.data)
print(output.size())
