_base_ = '../../default_runtime.py'
data_dir = '/home/user/data/nuscenes'
category_name = 'Trailer'
batch_size = 128
point_cloud_range = [-9.6, -9.6, -3.0, 9.6, 9.6, 3.0]
box_aware = True
use_rot = False


model = dict(
    type='P2PVoxel',
    backbone=dict(type='VoxelNet',
                  points_features=3,
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.15, 0.15, 0.3],
                  grid_size=[21, 128, 128],
                  output_channels=128
                  ),
    fuser=dict(type='BEVFuser'),
    head=dict(
        type='VoxelHead',
        q_distribution='laplace',  # ['laplace', 'gaussian']
        use_rot=use_rot,
        box_aware=box_aware
    ),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot
    )
)

train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='NuScenesDataset',
        path=data_dir,
        split='train_track',
        category_name=category_name,
        preloading=False,
        preload_offset=10,
    ),
    cfg=dict(
        num_candidates=4,
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        regular_pc=False,
        flip=True
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='NuScenesDataset',
        path=data_dir,
        split='val',
        category_name=category_name,
        preloading=False
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
