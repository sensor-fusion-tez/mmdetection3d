_base_ = [
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/datasets/kitti-3d-3class-nus.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/models/centerpoint_005voxel_second_secfpn_kitti_nus.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/schedules/cyclic_80e.py',
    '/home/alper/Desktop/Works/Tez/mmdetection3d/configs/_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
backend_args = None

# dataset settings
dataset_type = 'KittiDatasetNus'
data_root = 'data/kitti/'
#class_names = ['Car']
class_names = ['pedestrian', 'bicycle', 'car']

voxel_size = [0.05, 0.05, 0.1]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Add 'point_cloud_range' into model config according to dataset
voxel_size = [0.05, 0.05, 0.1]
model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            point_cloud_range = [0, -40, -3, 70.4, 40, 1],

            max_num_points=5,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),

    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['bicycle']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            pc_range=[-75.2, -75.2],
            type='CenterPointBBoxCoder',
            post_center_range=[-10, -50, -10, 80.4, 50, 10],
            max_num=100,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7,
        ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            pc_range=[-75.2, -75.2],
            point_cloud_range=point_cloud_range,
            grid_size=[1408, 1600, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-75.2, -75.2],
            point_cloud_range=point_cloud_range,
            post_center_limit_range=[-10, -50, -10, 80.4, 50, 10],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.2)))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(car=5, pedestrian=10, bicycle=10)),
    classes=class_names,
    sample_groups=dict(car=12, pedestrian=6, bicycle=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=16384),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4, dataset=dict(dataset=dict(pipeline=train_pipeline, )))

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
