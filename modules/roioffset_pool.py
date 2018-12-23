from torch.nn.modules.module import Module
from functions.roioffset_pool import RoIOffsetPoolFunction
import torch

class _RoIOffsetPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, use_offset=True, offset=None, trans_std=0.1):
        super(_RoIOffsetPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.use_offset = use_offset 
        self.trans_std = trans_std
           
    def forward(self, features, rois, offset=None):
        if not self.use_offset:
            offset = features.new(features.shape[0], 2, 7, 7).fill_(0.)

        offset = offset * self.trans_std
        
        return RoIOffsetPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois, offset)
