_base_ = [
    # '../../_base_/schedules/sgd_tsm_100e.py', 
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='Recognizer2D',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.5],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    backbone=dict(
        type='TANet',
        pretrained='ckpts/resnet18-5c106cde.pth',
        depth=18,
        num_segments=8,
        tam_cfg=None),
    cls_head=dict(
        type='TSMHead',
        num_classes=51,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/ma-user/work/data/hmdb51_org'
data_root_val = '/home/ma-user/work/data/hmdb51_org'
# data_root = '/home/ma-user/modelarts/inputs/hmdb51_org'
# data_root_val = '/home/ma-user/modelarts/inputs/hmdb51_org'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_videos.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_videos.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=8, num_clips=1, test_mode=False),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=8, num_clips=1, test_mode=True),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=8, num_clips=2, test_mode=True),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
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
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[12, 24],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=5, interval=5, save_best='auto'), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)

log_level = 'INFO'
load_from = 'ckpts/tanet_imagenet-pretrained-r18_8xb8-dense-1x1x8-100e_kinetics400-rgb_top1_epoch_100.pth'
resume = False