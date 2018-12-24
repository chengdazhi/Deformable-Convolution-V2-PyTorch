import torch
from torch.autograd import Variable
from dcn_v2 import DCNv2
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, '/data1/home/v-dachen/external/mxnet/mxnet_v1.1.0_dcnv2')
import mxnet as mx
from mxnet import autograd

N, inC, inH, inW = 2, 16, 64, 64
# N, inC, inH, inW = 2, 3, 1, 1
outC, outH, outW = 4, 64, 64
# outC = 3
kH, kW = 3, 3
# kH, kW = 1, 1
num_deformable_groups = 1

dcn_th = DCNv2(inC, outC, kH, 1, 1, 1, num_deformable_groups, no_bias=True)

weight = np.random.normal(0,1,(4,16,3,3)).astype(np.float32)
inputs = np.random.normal(0,1,(N, inC, inH, inW)).astype(np.float32)
mask = np.random.uniform(0, 1, (N, num_deformable_groups * kH * kW, outH, outW)).astype(np.float32)
offset = np.random.normal(0, 1, (N, num_deformable_groups * 2 * kH * kW, outH, outW)).astype(np.float32)

dcn_th.weight.data = torch.from_numpy(weight).cuda()
inputs_var = Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)
offset_var = Variable(torch.from_numpy(offset).cuda(), requires_grad=True)
mask_var = Variable(torch.from_numpy(mask).cuda(), requires_grad=True)

output_th = dcn_th(inputs_var, offset_var, mask_var)

weight_mx = mx.nd.array(weight, ctx=mx.gpu(0))
inputs_mx = mx.nd.array(inputs, ctx=mx.gpu(0))
offset_mx = mx.nd.array(offset, ctx=mx.gpu(0))
mask_mx = mx.nd.array(mask, ctx=mx.gpu(0))

weight_mx.attach_grad()
inputs_mx.attach_grad()
offset_mx.attach_grad()
mask_mx.attach_grad()

with autograd.record():
    output = mx.nd.contrib.ModulatedDeformableConvolution(data=inputs_mx, weight=weight_mx, offset=offset_mx,
                mask=mask_mx, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=outC, no_bias=True,
                num_deformable_group=1)

import IPython
IPython.embed()
