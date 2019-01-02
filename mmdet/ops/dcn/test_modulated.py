#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.modulated_dcn import ModulatedDeformConvPack
from modules.modulated_dcn import DeformRoIPooling
from modules.modulated_dcn import ModulatedDeformRoIPoolingPack

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3


def example_dconv():
    from modules.modulated_dcn import ModulatedDeformConv
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = ModulatedDeformConvPack(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

def example_dpooling():
    from modules.modulated_dcn import ModulatedDeformRoIPoolingPack
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(20, 2, 7, 7).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    # normal roi_align
    pooling = DeformRoIPooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=32,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1).cuda()

    # deformable pooling
    dpooling = DeformRoIPooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=32,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1).cuda()

    out = pooling(input, rois, offset)
    dout = dpooling(input, rois, offset)
    print(out.shape)
    print(dout.shape)

    target_out = out.new(*out.size())
    target_out.data.uniform_(-0.01, 0.01)
    target_dout = dout.new(*dout.size())
    target_dout.data.uniform_(-0.01, 0.01)
    e = (target_out - out).mean()
    e.backward()
    e = (target_dout - dout).mean()
    e.backward()

def example_mdpooling():
    from modules.modulated_dcn import ModulatedDeformRoIPoolingPack
    input = torch.randn(2, 32, 64, 64).cuda()
    input.requires_grad = True
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    dpooling = ModulatedDeformRoIPoolingPack(spatial_scale=1.0 / 4,
                         pooled_size=7,
                         output_dim=32,
                         no_trans=False,
                         group_size=1,
                         trans_std=0.1).cuda()

    for i in range(2):
        dout = dpooling(input, rois)
        target = dout.new(*dout.size())
        target.data.uniform_(-0.1, 0.1)
        error = (target - dout).mean()
        error.backward()
        print(dout.shape)

if __name__ == '__main__':

    example_dconv()
    example_dpooling()
    example_mdpooling()
