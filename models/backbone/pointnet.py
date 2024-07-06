from mmengine.registry import MODELS
import torch.nn as nn


@MODELS.register_module()
class PointNet(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.SyncBatchNorm(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, inputs):
        return super().forward(inputs.transpose(1, 2).contiguous())

