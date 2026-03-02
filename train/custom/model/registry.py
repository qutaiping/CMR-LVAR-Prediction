from mmcv.utils import Registry
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
PIPELINES = Registry('pipelines')
LOSSES = Registry('loss')
NETWORKS = Registry('network')