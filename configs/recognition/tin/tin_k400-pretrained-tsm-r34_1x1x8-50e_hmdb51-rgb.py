_base_ = [ 
    '../../_base_/schedules/sgd_50e.py',
    '../../_base_/default_runtime.py'
]
# model settings

preprocess_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], format_shape='NCHW')

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTIN',
        pretrained='ckpts/resnet34-333f7ec4.pth',
        depth=34,
        norm_eval=False,
        shift_div=4),
    cls_head=dict(
        type='TSMHead',
        num_classes=51,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None)

# model settings
# model = dict(cls_head=dict(is_shift=True))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/ma-user/work/data/hmdb51_org'
data_root_val = '/home/ma-user/work/data/hmdb51_org'
# data_root = '/home/ma-user/modelarts/inputs/hmdb51_org'
# data_root_val = '/home/ma-user/modelarts/inputs/hmdb51_org'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_videos.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_videos.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_videos.txt'

# sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
train_pipeline = [
    dict(type='DecordInit'),
    #  dict(type='UniformSample', clip_len=8, num_clips=1, test_mode=False),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    #  dict(type='UniformSample', clip_len=8, num_clips=1, test_mode=True),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    # dict(type='UniformSample', clip_len=8, num_clips=2, test_mode=True),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
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
    num_workers=2,
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

train_cfg = dict(val_interval=2)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[12, 24],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor', paramwise_cfg=dict(fc_lr5=True))
#参考了mmaction2官方训练日志
auto_scale_lr = dict(enable=True, base_batch_size=48)
load_from = 'ckpts/tin_kinetics400-pretrained-tsm-r34_1x1x8-50e_kinetics400-rgb_top1_epoch_45.pth'#'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'  # noqa: E501
