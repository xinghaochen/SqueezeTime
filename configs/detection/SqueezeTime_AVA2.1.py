_base_ = '../../_base_/default_runtime.py'

# url = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
#        'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-'
#        'rgb/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_'
#        'kinetics400-rgb_20220901-e7b65fad.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    # init_cfg=dict(type='Pretrained', checkpoint=url),
    # backbone=dict(
    #     type='mmaction.ResNet3dSlowOnly',
    #     depth=50,
    #     pretrained=url,
    #     pretrained2d=False,
    #     lateral=False,
    #     num_stages=4,
    #     conv1_kernel=(1, 7, 7),
    #     conv1_stride_t=1,
    #     pool1_stride_t=1,
    #     spatial_strides=(1, 2, 2, 1)),
    backbone=dict(
        type='mmaction.SqueezeTime_ava',
        #pretrained2d=False,
        #pretrained='ckpts/res2d_v6_5_w1.pth',
        depth=50,widen_factor=1.0,dropout=0.5,input_channels=48,n_classes=400,load='configs/recognition/SSqueezeTime/SqueezeTime_K400.py',freeze_bn=False,spatial_stride=[1,2,2,1],pos_dim=[56,28,14,7]),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=128,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
# data_root = "/home/ma-user/work/data/ava/rawframes" #'data/ava/rawframes'
data_root= "/home/ma-user/modelarts/inputs/rawframes"
anno_root = 'data/ava'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
ann_file_val = f'{anno_root}/ava_val_v2.1.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'

label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 288)),
    dict(
        type='MultiScaleCrop',
        input_size=256,
        scales=(1, 0.875, 0.75, 0.66, 0.5),
        random_crop=True,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    # dict(type='RandomRescale', scale_range=(256, 320)),
    # dict(type='RandomCrop', size=256),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=16, frame_interval=4, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(256, 256),keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr=1.0
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1 / 10,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=5,
        end=50,
        convert_to_iter_based=True)
]


optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=128)

#best 63.8