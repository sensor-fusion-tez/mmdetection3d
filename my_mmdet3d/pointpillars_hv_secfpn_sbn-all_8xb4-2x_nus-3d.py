_base_ = [
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/models/pointpillars_hv_fpn_nus.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/schedules/schedule-2x.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/default_runtime.py',
]

class_names = ['pedestrian', 'bicycle','car']
metainfo = dict(_delete_=True, classes=class_names)

# model settings
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],


            ],
            sizes=[
                [0.7256437, 0.66344886, 1.75748069],  # pedestrian
                [1.68452161, 0.60058911, 1.27192197],  # bicycle
                [4.60718145, 1.95017717, 1.72270761],  # car


            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=5),
)