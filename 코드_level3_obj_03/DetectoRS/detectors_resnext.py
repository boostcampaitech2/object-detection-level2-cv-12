
_base_ = 'detectors_cascade_rcnn_r101_1x_coco.py'

epoch = 50

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='object_detection',
                name='eric9687')
        )
    ])

custom_hooks = [dict(type='NumClassCheckHook')]
workflow = [('train', 1)]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/opt/ml/code/mmdetection_trash/work_dirs/detectors_v5/best_bbox_mAP_50.pth'
resume_from = None

# optimizer
optimizer = dict(type='AdamW', lr=0.00005, weight_decay=5e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
total_iters = 1000000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[250000, 250000, 250000, 250000],
    restart_weights=[1, 1, 1, 1],
    min_lr=1e-7)

# lr_config = dict(
#     policy='CosineRestart',
#     # warmup='linear',    
#     # warmup_iters=1500,
#     # warmup_ratio=1.0 / 10,
#     by_epoch=False,
#     periods=[3000, 3000, 3000, 3000, 3000,3000, 3000]*7,
#     restart_weights=[1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05] * 7,
#     min_lr=1e-4)



# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=0.25)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch)
# evaluation = dict(interval=1, metric=['bbox'])
evaluation = dict(
    interval=1,
    save_best='bbox_mAP_50',
    metric=['bbox']
)

# max-depth 3
checkpoint_config = dict(max_keep_ckpts=3, interval=1)