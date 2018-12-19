import torch
from torch.autograd import Variable
from modules import ConvOffset2d
import torch.nn as nn
import numpy as np

#N, inC, inH, inW = 256, 16, 64, 64
N, inC, inH, inW = 2, 3, 1, 1
#outC = 4
outC = 3
#kH, kW = 3, 3
kH, kW = 1, 1
num_deformable_groups = 1

conv = nn.Conv2d(
    inC,
    outC,
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=((kW-1)//2, (kW-1)//2),
    bias=False).cuda()


conv_offset2d = ConvOffset2d(
    inC,
    outC, (kH, kW),
    stride=1,
    padding=(kW-1)//2,
    num_deformable_groups=num_deformable_groups).cuda()

weight = torch.from_numpy(np.arange(9).astype(np.float32)).reshape((3,3,1,1,)).cuda()

#weight = torch.from_numpy(np.random.normal(0,1,(4,16,3,3)).astype(np.float32)).cuda()

conv_offset2d.weight.data = weight
conv.weight.data = weight

inputs = np.arange(N * inC * inH * inW).reshape((N, inC, inH, inW)).astype(np.float32)
#inputs = np.random.normal(0,1,(N, inC, inH, inW)).astype(np.float32)

inputs_var1 = Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)

inputs_var2 = Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)

offset = Variable(torch.zeros(N, num_deformable_groups * 2 * kH * kW, inH, inW).cuda(), requires_grad=True)

output1 = conv(inputs_var1)

output2 = conv_offset2d(inputs_var2, offset)

import IPython
IPython.embed()
