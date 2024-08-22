_base_ = ['../configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py']
num_things_classes = 37 #80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2Former',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    panoptic_head=dict(
        type='Mask2FormerHeadMutilHeadTwoModule3Small2',
        img_size=(512,512),
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            encoder=dict(
                type='DetrTransformerDecoder'
            )),
        transformer_decoder = dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                num_queries=100,
                prior_attn_num_heads=2,
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),

            init_cfg=None),

        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
img_scale_train = [(512,512)] 
img_scale_test = (512,512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize', img_scale=img_scale_train, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32), 
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'TreeSpeciesDataset'
imgs_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/'
data_root = '/media/ld/新加卷/zhulirong/data/tree_speices_cuted_20220722/Annotations/'

data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_treespecies_train_2023.json',
        img_prefix=imgs_root + 'image',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
                 'instances_treespecies_test_2023.json',
        img_prefix=imgs_root + 'image',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_treespecies_test_2023.json',
        img_prefix=imgs_root + 'image',
        pipeline=test_pipeline))


embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

max_iters = 368750
runner = dict(type='IterBasedRunner', max_iters=max_iters)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
interval = 5000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3)

dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict(
    interval=interval,
    dynamic_intervals=dynamic_intervals,
    metric=['bbox', 'segm'],
    save_best='auto')

work_dir = './work_dirs/mask2former_r50_treesspecies_edition_twomodule2/small_lr0.0001_50e_1108'
load_from = './checkpoints/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

