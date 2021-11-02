_base_ = [
     '../_base_/datasets/caltech256_bs64.py', '../_base_/default_runtime.py'
]

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=257),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


# load model pretrained on imagenet
load_from = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth'  # noqa

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
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
