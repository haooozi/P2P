_base_ = '../../default_runtime.py'
data_dir = '/home/user/den/data/kitti/my_kitti'
category_name = 'Van'
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

train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Train',
        category_name=category_name,
        preloading=True,
        preload_offset=10
    ),
    cfg=dict(
        num_candidates=4,
        target_thr=10,
        search_thr=20,
        point_cloud_range=point_cloud_range,
        regular_pc=True,
        flip=False
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Test',
        category_name=category_name,
        preloading=True,
        preload_offset=-1
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
