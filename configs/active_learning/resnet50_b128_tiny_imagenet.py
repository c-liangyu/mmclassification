_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/tiny_imagenet_bs128.py',
    '../_base_/default_runtime.py'
]
model = dict(head=dict(num_classes=200))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=100)

checkpoint_config = dict(interval=20)


