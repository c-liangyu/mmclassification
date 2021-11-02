_base_ = [
     '../_base_/datasets/caltech256_bs64.py', '../_base_/default_runtime.py', '../_base_/models/resnet50.py',
]

# load model pretrained on imagenet
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=257),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=30, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=40)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=10)
