from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn

from mmdet3d.models.layers import build_sa_module
from mmdet3d.utils import ConfigType
from mmengine.registry import MODELS


@MODELS.register_module()
class PointNet2(nn.Module):
    """PointNet2 with Single-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_points: Sequence[int] = (512, 256, 128),
                 radius: Sequence[float] = (0.3, 0.5, 0.7),
                 num_samples: Sequence[int] = (32, 32, 32),
                 sa_channels: Sequence[Sequence[int]] = ((64, 64, 128),
                                                         (128, 128, 256),
                                                         (256, 256, 256)),
                 norm_cfg: ConfigType = dict(type='SyncBN', eps=1e-3, momentum=0.01),
                 sa_cfg: ConfigType = dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True)):
        super().__init__()
        self.num_sa = len(sa_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            sa_in_channel = sa_out_channel

        self.FA_module = nn.Sequential(
            nn.Conv1d(sa_in_channel + 3, 1024, 1, bias=False),
            nn.SyncBatchNorm(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
        )

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def forward(self, points: Tensor) -> Dict[str, List[Tensor]]:
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of
                    each fp features.
                - fp_features (list[torch.Tensor]): The features
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        feat = self.FA_module(
            torch.cat([sa_xyz[-1].transpose(1, 2).contiguous(), sa_features[-1]], 1))

        return feat
