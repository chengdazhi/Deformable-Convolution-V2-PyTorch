#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair

from functions.deform_psroi_pooling_func import DeformRoIPoolingFunction

class DeformRoIPooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return DeformRoIPoolingFunction.apply(input, rois, offset,
                                              self.spatial_scale,
                                              self.pooled_size,
                                              self.output_dim,
                                              self.no_trans,
                                              self.group_size,
                                              self.part_size,
                                              self.sample_per_part,
                                              self.trans_std)

_DeformRoIPooling = DeformRoIPoolingFunction.apply

class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale,
                                         pooled_size,
                                         output_dim,
                                         no_trans,
                                         group_size,
                                         part_size,
                                         sample_per_part,
                                         trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size *
                          self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size *
                          self.pooled_size * 3)
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = DeformRoIPoolingFunction.apply(input, rois, offset,
                                                 self.spatial_scale,
                                                 self.pooled_size,
                                                 self.output_dim,
                                                 True,  # no trans
                                                 self.group_size,
                                                 self.part_size,
                                                 self.sample_per_part,
                                                 self.trans_std)

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(
                n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return DeformRoIPoolingFunction.apply(input, rois, offset,
                                                  self.spatial_scale,
                                                  self.pooled_size,
                                                  self.output_dim,
                                                  self.no_trans,
                                                  self.group_size,
                                                  self.part_size,
                                                  self.sample_per_part,
                                                  self.trans_std) * mask
        # only roi_align
        return DeformRoIPoolingFunction.apply(input, rois, offset,
                                              self.spatial_scale,
                                              self.pooled_size,
                                              self.output_dim,
                                              self.no_trans,
                                              self.group_size,
                                              self.part_size,
                                              self.sample_per_part,
                                              self.trans_std)
