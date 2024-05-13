_base_ = [
    # '../../_base_/models/tpn_slowonly_r18.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=18,
        pretrained='ckpts/resnet18-5c106cde.pth',
        lateral=False,
        out_indices=(2, 3),
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    neck=dict(
        type='TPN',
        in_channels=(256, 512),
        out_channels=256,
        spatial_modulation_cfg=dict(
            in_channels=(256, 512), out_channels=512),
        temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(256, 256),
            mid_channels=(256, 256),
            out_channels=512,
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=600, loss_weight=0.5)),
    cls_head=dict(
        type='TPNHead',
        num_classes=600,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(fcn_test=True))
dataset_type = 'VideoDataset'
# data_root = '/home/ma-user/work/data/k600/Kinetics600/videos'
# data_root_val = '/home/ma-user/work/data/k600/Kinetics600/videos'
data_root = '/home/ma-user/modelarts/inputs/Kinetics600/videos'
data_root_val = '/home/ma-user/modelarts/inputs/Kinetics600/videos'
ann_file_train = 'data/kinetics600/kinetics600_train_list_videos.txt'
ann_file_val = 'data/kinetics600/kinetics600_val_list_videos.txt'
ann_file_test = 'data/kinetics600/kinetics600_val_list_videos.txt'
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True),
    clip_grad=dict(max_norm=40, norm_type=2),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[75, 125],
        gamma=0.1)
]
auto_scale_lr = dict(enable=True, base_batch_size=64)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))
