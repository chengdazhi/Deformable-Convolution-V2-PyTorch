#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN

class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN.modulated_deform_conv_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.groups,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            DCN.modulated_deform_conv_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.groups,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None, None, None
