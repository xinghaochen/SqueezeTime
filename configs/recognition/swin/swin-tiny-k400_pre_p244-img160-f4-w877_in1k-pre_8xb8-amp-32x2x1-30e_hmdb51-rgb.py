_base_ = [
     '../../_base_/default_runtime.py'
]
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained='ckpts/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=51,
        spatial_type='avg',
        dropout_ratio=0.5,
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
    dict(type='SampleFrames', clip_len=4, frame_interval=12, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(160, 160), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=12,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 160)),
    dict(type='ThreeCrop', crop_size=160),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=12,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 160)),
    dict(type='ThreeCrop', crop_size=160),
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
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)

log_level = 'INFO'
load_from = 'ckpts/swin-tiny-p244-img160-f4-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400_top1_epoch_30.pth'
resume = False