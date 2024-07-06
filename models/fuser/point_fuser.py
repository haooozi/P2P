import torch
import torch.nn as nn
from mmengine.registry import MODELS


class MLPMixer(nn.Sequential):

    def __init__(self, in_channels, out_channels, embed_dim=1024):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.SyncBatchNorm(embed_dim, eps=1e-3, momentum=0.01),
        )

    def forward(self, inputs):
        return super().forward(inputs)


@MODELS.register_module()
class PointFuser(nn.Module):

    def __init__(self, box_aware):
        super().__init__()
        self.box_aware = box_aware
        self.fuse = nn.Sequential(
            MLPMixer(3 if box_aware else 2, 64),
            MLPMixer(64, 128),
            MLPMixer(128, 256),
            nn.Linear(256, 1),
            nn.SyncBatchNorm(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Flatten(),
        )
        if box_aware:
            self.wlh_mlp = nn.Sequential(
                nn.Linear(3, 128),
                nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(128, 1024)
            )

    def forward(self, stack_feats, wlh=None):
        prev_feats, this_feats = torch.split(stack_feats, stack_feats.size(0) // 2, 0)
        if self.box_aware and wlh is not None:
            wlh = self.wlh_mlp(wlh).unsqueeze(-1)
            cat_feats = torch.cat([prev_feats, this_feats, wlh], 2)
        else:
            cat_feats = torch.cat([prev_feats, this_feats], 2)
        return self.fuse(cat_feats)
