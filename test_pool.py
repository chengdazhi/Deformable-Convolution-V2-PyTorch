import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from modules import _RoIOffsetPooling

num_deformable_groups = 2

N, inC, inH, inW = 2, 6, 32, 32

deform_roi_pooling = _RoIOffsetPooling(7, 7, 0.0625, True, trans_std=0.1)

inputs_np = np.random.randn(N, inC, inH, inW).astype(np.float32)
inputs = Variable(torch.from_numpy(inputs_np).cuda(), requires_grad=True)

rois_np = np.array([[0, 10, 20, 300, 400], [0, 50, 10, 90, 200], [1, 100, 200, 400, 300], [1, 500.3, 0.4, 510.9, 511.0]]).astype(np.float32)
rois = Variable(torch.from_numpy(rois_np).cuda())

#offset_np = np.random.randn(3, 7, 7, 2).astype(np.float32)
offset_np = np.zeros((4, 7, 7, 2)).astype(np.float32)
offset = Variable(torch.from_numpy(offset_np).cuda(), requires_grad=True)

output = deform_roi_pooling(inputs, rois, offset)
import IPython
IPython.embed()

exit()
#output.backward(output.data)

import mxnet as mx
from mxnet import autograd

inputs_mx = mx.nd.array(inputs_np, ctx=mx.gpu(0))
inputs_mx.attach_grad()

rois_mx = mx.nd.array(rois_np, ctx=mx.gpu(0))
rois_mx.attach_grad()

offset_mx = mx.nd.array(offset_np, ctx=mx.gpu(0)).transpose((0,3,1,2))
offset_mx.attach_grad()

with autograd.record():
    output_mx = mx.nd.contrib.DeformablePSROIPooling(data=inputs_mx, rois=rois_mx,
        trans=offset_mx, group_size=1, pooled_size=7, sample_per_part=4,
        no_trans=False, part_size=7, output_dim=inC, spatial_scale=0.0625, trans_std=0.1)

import IPython
IPython.embed()
