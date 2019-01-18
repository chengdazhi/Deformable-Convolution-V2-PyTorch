#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from modules.modulated_deform_conv import ModulatedDeformConv, _ModulatedDeformConv, ModulatedDeformConvPack
from modules.deform_psroi_pooling import DeformRoIPooling, _DeformRoIPooling, DeformRoIPoolingPack

deformable_groups = 1
N, inC, inH, inW = 2, 4, 4, 4
outC = 4
kH, kW = 3, 3


torch.manual_seed(3)
def conv_identify(weight, bias, groups=1):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    oc = o // groups
    for p in range(i):
        for q in range(o):
            if (p) == (q % oc):
                # print(q, p, y, x)
                # print(q % oc)
                weight.data[q, p, y, x] = 1.0

def check_dconv_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    dcn = DeformConv(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   groups=2, 
                   deformable_groups=deformable_groups, im2col_step=1).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW), stride=1, padding=1, dilation=1, groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    # conv_identify(dcn.weight, dcn.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output_d = dcn(input, offset)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv zero offset passed with {}'.format(d))
    else:
        print('dconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())

def check_mdconv_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    dcn = ModulatedDeformConv(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   groups=2, 
                   deformable_groups=deformable_groups, im2col_step=1).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW), stride=1, padding=1, dilation=1, groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    mask *= 2
    output_d = dcn(input, offset, mask)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('mdconv zero offset passed with {}'.format(d))
    else:
        print('mdconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())

def check_dconv_zero_offset_identify():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    groups = 2
    dcn = DeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=1).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output = dcn(input, offset)
    d = (input - output).abs().max()
    if d < 1e-10:
        print('dconv zero offset identify passed with {}'.format(d))
    else:
        print('dconv zero offset identify failed with {}'.format(d))
        # print(input)
        # print(output)
        print((input - output).abs())

def check_mdconv_zero_offset_identify():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    groups = 2
    dcn = ModulatedDeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=1).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('mdconv zero offset identify passed with {}'.format(d))
    else:
        print('mdconv zero offset identify failed with {}'.format(d))
        # print(input)
        # print(output)
        print((input - output).abs())

def check_dconv_im2col_step_forward():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    groups = 2

    dcn1 = DeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=1).cuda()

    dcn2 = DeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=2).cuda()
    dcn1.weight = dcn2.weight
    dcn1.bias = dcn2.bias
    output1 = dcn1(input, offset)
    output2 = dcn2(input, offset)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('dconv im2col_step forward passed with {}'.format(d))
    else:
        print('dconv im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())

def check_mdconv_im2col_step_forward():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    groups = 2

    dcn1 = ModulatedDeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=1).cuda()

    dcn2 = ModulatedDeformConv(inC, outC, (kH, kW), 
        stride=1, padding=1, dilation=1, 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=2).cuda()
    dcn1.weight = dcn2.weight
    dcn1.bias = dcn2.bias
    output1 = dcn1(input, offset, mask)
    output2 = dcn2(input, offset, mask)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('mdconv im2col_step forward passed with {}'.format(d))
    else:
        print('mdconv im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())

def check_dconv_im2col_step_backward():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _DeformConv(input, offset, weight, bias, stride, padding, dilation, groups, deformable_groups, 2)
    targert = torch.rand(*output1.size()).cuda()
    error = (targert - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _DeformConv(input, offset, weight, bias, stride, padding, dilation, groups, deformable_groups, 1)
    error2 = (targert - output2).mean()
    error.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max() 
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + weight_grad_err + bias_grad_err
    if grad_err:
        print("dconv im2col_step backward passed with {} = {}+{}+{}+{}".format(grad_err, input_grad_err, offset_grad_err, weight_grad_err, bias_grad_err))
    else:
        print("dconv im2col_step backward failed with {} = {}+{}+{}+{}".format(grad_err, input_grad_err, offset_grad_err, weight_grad_err, bias_grad_err))

def check_mdconv_im2col_step_backward():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.sigmoid(torch.randn(N, deformable_groups * 1 * kW * kH, inH, inW).cuda())
    mask.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _ModulatedDeformConv(input, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups, 2)
    targert = torch.rand(*output1.size()).cuda()
    error = (targert - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    mask_grad = mask.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _ModulatedDeformConv(input, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups, 1)
    error2 = (targert - output2).mean()
    error.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max() 
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    mask_grad_err = (mask.grad - 2 * mask_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + mask_grad_err + weight_grad_err + bias_grad_err
    if grad_err < 1e-7:
        print("mdconv im2col_step backward passed with {}".format(grad_err))
    else:
        print("mdconv im2col_step backward failed with {}".format(grad_err))

def check_gradient_conv():

    input = torch.rand(N, inC, inH, inW).double().cuda() * 0.01
    input.requires_grad = True
    from torch.nn.functional import conv2d

    weight = torch.randn(outC, inC, kH, kW).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    stride = 1
    padding = 1
    dilation = 1

    # print('check_gradient_conv: ',
    #       gradcheck(conv2d, (input, weight, bias,
    #                 stride, padding, dilation, deformable_groups),
    #                 eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True))
    print('check_gradient_conv: ',
          gradcheck(conv2d, (input, weight, bias,
                    stride, padding, dilation, deformable_groups)))

def check_gradient_dconv():

    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(N, inC, inH, inW).double().cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).double().cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), kH, kW).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    print('check_gradient_dconv: ',
          gradcheck(_DeformConv, (input, offset, weight, bias,
                    stride, padding, dilation, groups, deformable_groups, im2col_step),
                    eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))
    # print('check_gradient_dconv: ',
    #       gradcheck(_DeformConv, (input, offset, weight, bias,
    #                 stride, padding, dilation, deformable_groups)))

def check_gradient_mdconv():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, int(inC//groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    print('check_gradient_mdconv: ',
          gradcheck(_ModulatedDeformConv, (input, offset, mask, weight, bias,
                    stride, padding, dilation, groups, deformable_groups, im2col_step),
                    eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))


def check_pooling_zero_offset():

    input = torch.randn(2, 16, 64, 64).cuda().zero_()
    input[0, :, 16:26, 16:26] = 1.
    input[1, :, 10:20, 20:30] = 2.
    rois = torch.tensor([
        [0, 65, 65, 103, 103],
        [1, 81, 41, 119, 79],
    ]).cuda().float()
    pooling = DeformRoIPooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=16,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.0).cuda()

    out = pooling(input, rois, input.new())
    s = ', '.join(['%f' % out[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)

    dpooling = DeformRoIPooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=16,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.0).cuda()
    offset = torch.randn(20, 2, 7, 7).cuda().zero_()
    dout = dpooling(input, rois, offset)
    s = ', '.join(['%f' % dout[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)


def check_gradient_dpooling():
    input = torch.randn(2, 3, 5, 5).cuda() * 0.01
    N = 4
    batch_inds = torch.randint(2, (N, 1)).cuda().float()
    x = torch.rand((N, 1)).cuda().float() * 15
    y = torch.rand((N, 1)).cuda().float() * 15
    w = torch.rand((N, 1)).cuda().float() * 10
    h = torch.rand((N, 1)).cuda().float() * 10
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(N, 2, 3, 3).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    spatial_scale = 1.0 / 4
    pooled_size = 3
    output_dim = 3
    no_trans = 0
    group_size = 1
    trans_std = 0.0
    sample_per_part = 4
    part_size = pooled_size

    print('check_gradient_dpooling:',
          gradcheck(_DeformRoIPooling, (input, rois, offset,
                                     spatial_scale,
                                     pooled_size,
                                     output_dim,
                                     no_trans,
                                     group_size,
                                     part_size,
                                     sample_per_part,
                                     trans_std),
                    eps=1e-4))


def example_dconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DeformConvPack(64, 128, kernel_size=(3, 3), stride=1,
              padding=1, groups=2, deformable_groups=2).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

def example_mdconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = ModulatedDeformConvPack(64, 128, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)


def example_dpooling():
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
    input = torch.randn(2, 32, 64, 64).cuda()
    input.requires_grad = True
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    dpooling = DeformRoIPoolingPack(spatial_scale=1.0 / 4,
                          pooled_size=7,
                          output_dim=32,
                          no_trans=False,
                          group_size=1,
                          trans_std=0.1,
                          deform_fc_dim=1024).cuda()

    dout = dpooling(input, rois)
    target = dout.new(*dout.size())
    target.data.uniform_(-0.1, 0.1)
    error = (target - dout).mean()
    error.backward()
    print(dout.shape)


if __name__ == '__main__':

    example_dconv()
    example_mdconv()
    example_dpooling()
    example_mdpooling()

    print('checking')
    check_dconv_im2col_step_forward()
    check_dconv_im2col_step_backward()
    check_mdconv_im2col_step_forward()
    check_mdconv_im2col_step_backward()
    check_pooling_zero_offset()
    # zero offset check
    if inC == outC:
        check_dconv_zero_offset()
        check_dconv_zero_offset_identify()
        check_mdconv_zero_offset()
        check_mdconv_zero_offset_identify()

    check_gradient_conv()
    check_gradient_dconv()
    check_gradient_mdconv()
    check_gradient_dpooling()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
