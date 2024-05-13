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
        num_classes=400,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=False,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None)
