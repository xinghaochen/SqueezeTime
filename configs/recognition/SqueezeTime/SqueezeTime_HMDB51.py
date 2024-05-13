
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SqueezeTime',
        #pretrained2d=False,
        #pretrained='ckpts/res2d_v6_5_w1.pth',
        depth=50,widen_factor=1.0,dropout=0.5,input_channels=48,n_classes=400,load='ckpts/SqueezeTime_K400_71.64.pth',freeze_bn=False,pos_dim=[56,28,14,7]),
    cls_head=dict(
        type='I2DHead',
        num_classes=51,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # blending=dict(
        #     type='RandomBatchAugment',
        #     augments=[
        #         dict(type='MixupBlending', alpha=0.8, num_classes=51),
        #         dict(type='CutmixBlending', alpha=1, num_classes=51)
        #     ]),
        format_shape='NCTHW'))
# sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/hmdb51_org'
data_root_val = 'data/hmdb51_org'
# data_root = '/home/ma-user/modelarts/inputs/hmdb51_org'
# data_root_val = '/home/ma-user/modelarts/inputs/hmdb51_org'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_videos.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_videos.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66, 0.5),
        random_crop=True,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='ImgAug', transforms=[
    #             dict(type='Rotate', rotate=(-15, 15))
    #         ]),
    # dict(
    #     type='PytorchVideoWrapper',
    #     op='RandAugment',
    #     magnitude=7,
    #     num_layers=4),
    dict(type='ColorJitter'),
    dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16, num_clips=2, test_mode=True),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=2,test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16, num_clips=2, test_mode=True),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=2,test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 286)),
    dict(type='ThreeCrop', crop_size=286),
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
    batch_size=16,
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

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)
default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = 'ckpts/res2d_v6_19_w1_27_3_fast_v2_10_poswh_sum224_true_epoch_100_71.64.pth'
resume = False

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr=0.02
#0.0045 top1: 62.42 88.82
#lr 0.01 wd:0.0005 lr 6,16, top1 63.07
#lr 0.015 wd: 0.005 lr 12,30 backbone lrmulti 0.1 top1 63.27/63.73 top5 89.41/89.22
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1 / 10,
#         by_epoch=True,
#         begin=0,
#         end=2.5,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=27.5,
#         eta_min=base_lr / 100,
#         by_epoch=True,
#         begin=2.5,
#         end=30,
#         convert_to_iter_based=True)
# ]
param_scheduler = [
    # dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[12,24],
        gamma=0.1)
]
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.008),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1,decay_mult=1.0),
        }),
    clip_grad=dict(max_norm=40, norm_type=2))
#top1: 65.56, top5: 90.26, 12,24,lr 0.015, wd 0.008