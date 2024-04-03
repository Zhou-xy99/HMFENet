r""" Extracts intermediate features from given backbone network & layer ids """
from collections import Counter
import numpy as np

def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    layer_nums = np.cumsum(list(Counter(lids).values()))  # [ 3 7 13 16]
    layer_nums_iter = iter(layer_nums)
    layer_id = next(layer_nums_iter)
    layers = [feat]

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:#把layer1过完，从layer2开始保留
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        if hid + 1 == layer_id:#[ 3  7 13 16]保留了每一层layer最后一个bottleneck的输出
            if layer_id != layer_nums[-1]:
                layer_id = next(layer_nums_iter)
            layers.append(feat.clone())

    return feats, layers