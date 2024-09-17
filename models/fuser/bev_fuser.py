import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.registry import MODELS


@MODELS.register_module()
class BEVFuser(nn.Module):

    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.conv = nn.Sequential(
            # fuse
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(256, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(512, 1024, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):
        B, C, H, W = stack_feats.size()
        prev_feats, this_feats = torch.split(stack_feats, B // 2, 0)
        feats = torch.cat((prev_feats, this_feats), 1)

        return self.conv(feats)
