_base_ = ['slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb.py']

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=18,
        pretrained='ckpts/resnet18-5c106cde.pth',
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=51,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))
# sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
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
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
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
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))


log_level = 'INFO'
load_from = 'ckpts/slowonly_imagenet-pretrained-r18_8xb16-8x8x1-steplr-150e_kinetics400-rgb_top1_epoch_150.pth'
resume = False

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    # dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[12, 24],
        gamma=0.1)
]
