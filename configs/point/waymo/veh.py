_base_ = '../../default_runtime.py'
data_dir = '/home/user/den/waymo/'
category_name = 'Vehicle'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True
use_rot = False

model = dict(
    type='SATrackPoint',
    backbone=dict(type='PointNet'),
    fuser=dict(
        type='PointFuser',
        box_aware=box_aware,
    ),
    head=dict(
        type='PointHead',
        q_distribution='gaussian',  # ['laplace', 'gaussian']
        use_rot=use_rot
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
