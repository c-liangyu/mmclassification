_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/caltech256_bs64.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=100)

checkpoint_config = dict(interval=20)

