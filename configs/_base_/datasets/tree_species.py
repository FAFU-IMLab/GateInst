image_size = (512,512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=False),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset_type = 'TreeSpeciesDataset'
# imgs_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/'
# data_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/Annotations/'
#
# data = dict(
#     # _delete_=True,
#     samples_per_gpu=1,
#     workers_per_gpu=0,
#     train=dict(
#         type='RepeatDataset',
#         times=8,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=data_root + 'instances_treespecies_train_0808.json',
#             img_prefix=imgs_root + '0808_images',
#             pipeline=train_pipeline)),
#     # train=dict(
#     #     type=dataset_type,
#     #     ann_file=data_root + 'instances_treespecies_train_0808.json',
#     #     img_prefix=imgs_root + '0808_images',
#     #     pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root +
#                  'instances_treespecies_train_0801.json',
#         img_prefix=imgs_root + '0801_images',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'instances_treespecies_train_0801.json',
#         img_prefix=imgs_root + '0801_images',
#         pipeline=test_pipeline))
dataset_type = 'TreeSpeciesDataset'
data_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/Annotations/'
data_image_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_treespecies_train_0808.json',
        img_prefix=data_image_root + '0808_images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_treespecies_train_0801.json',
        img_prefix=data_image_root + '0801_images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_treespecies_train_0801.json',
        img_prefix=data_image_root + '0801_images',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])