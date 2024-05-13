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
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))
dataset_type = 'VideoDataset'
# data_root = '/home/ma-user/work/data/kinetics400/train_256'
# data_root_val = '/home/ma-user/work/data/kinetics400/val_256'
data_root = '/home/ma-user/modelarts/inputs/kinetics_videos/train_256'
data_root_val = '/home/ma-user/modelarts/inputs/kinetics_videos/val_256'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=10,
        end=150,
        by_epoch=True,
        milestones=[90, 130],
        gamma=0.1)
]
