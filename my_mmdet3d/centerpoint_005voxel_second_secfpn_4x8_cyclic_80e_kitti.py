_base_ = [
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/models/centerpoint_005voxel_second_secfpn_kitti.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/schedules/cyclic-40e.py', 
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Add 'point_cloud_range' into model config according to dataset
model = dict(
        data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=10),
    logger=dict(interval=1, type='LoggerHook'))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # TensorBoard için aşağıdaki satırı ekleyebilirsiniz.
        # dict(type='TensorboardLoggerHook')
    ])

load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = True  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.