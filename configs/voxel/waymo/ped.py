_base_ = '../../default_runtime.py'
data_dir = '/home/user/den/waymo/'
category_name = 'Pedestrian'
batch_size = 128
point_cloud_range = [-1.92, -1.92, -1.5, 1.92, 1.92, 1.5]
box_aware = False
use_rot = False

model = dict(
    type='SATrackVoxel',
    backbone=dict(type='VoxelNet',
                  points_features=3,
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.075, 0.075, 0.15],
                  grid_size=[21, 128, 128],
                  output_channels=128
                  ),
    fuser=dict(type='BEVFuser'),
    head=dict(
        type='VoxelHead',
        q_distribution='gaussian',  # ['laplace', 'gaussian']
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

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='WaymoDataset',
        path=data_dir,
        category_name=category_name,
        mode='all'
    ),
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
