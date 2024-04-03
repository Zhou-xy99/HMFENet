r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from common.olm4 import olm
import numpy as np
from .base.CRM import NonBottleneck1D

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

class HypercorrSqueezeNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HypercorrSqueezeNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone #'resnet50'
        self.use_original_imgsize = use_original_imgsize#false
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]#无用
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))#[0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])#[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]#tensor([ 3,  9, 13])
        self.backbone.eval()

        self.em4 = olm(64)

        self.gam = Attention(in_channels=256)
        self.attn1 = NonBottleneck1D(256, 256)
        reduce_dim = 256
        self.low_fea_id = 'layer2'
        if self.backbone_type=="vgg16":
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 64 + 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))#3,6,4
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Conv2d(256, 64, (3, 3), padding=(1, 1), bias=True), #256
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.1),
                                      nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))


    def forward(self, query_img,query_img_th, support_img,support_img_th,support_mask,history_mask):
        supps = []
        supp_ths=[]
        quys = []
        quy_ths=[]
        corrs=[]
        supp_pro_list = []
        supp_pro_list1 = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        gams = 0

        with torch.no_grad():
            query_feats,query_backbone_layers = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            if True:
                query_backbone_layers[3] = F.interpolate(query_backbone_layers[3],
                                           size=(query_backbone_layers[2].size(2), query_backbone_layers[2].size(3)), \
                                           mode='bilinear', align_corners=True)
                query_backbone_layers[4] = F.interpolate(query_backbone_layers[4],
                                           size=(query_backbone_layers[2].size(2), query_backbone_layers[2].size(3)), \
                                           mode='bilinear', align_corners=True)

                query_feat0 = torch.cat([query_backbone_layers[3], query_backbone_layers[2]],1)

            query_feat0 = self.down_query(query_feat0)

            quy = self.attn1(query_feat0)


            support_feats,support_backbone_layers = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            final_supp_list.append(support_backbone_layers[4])

            if True:
                support_backbone_layers[3] = F.interpolate(support_backbone_layers[3], size=(support_backbone_layers[2].size(2),support_backbone_layers[2].size(3)),
                                            mode='bilinear', align_corners=True)
                support_backbone_layers[4] = F.interpolate(support_backbone_layers[4],
                                          size=(support_backbone_layers[4].size(2), support_backbone_layers[4].size(3)),
                                          mode='bilinear', align_corners=True)





            # gam= self.gam(supp_feat, support_mask)#torch.Size([2, 256, 25, 25]),torch.Size([2, 200, 200])
            supp_feat_list.append(support_backbone_layers[2])




            support_feats = self.mask_feature(support_feats, support_mask.clone())

            query_feats_th,query_backbone_layers_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th,support_backbone_layers_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.mask_feature(support_feats_th, support_mask.clone())

            ##################################################################################################################################
            support_layer3 = F.interpolate(support_feats[9], size=(support_feats[3].size(2), support_feats[3].size(3)),
                                           mode='bilinear', align_corners=True)
            supp_feat = torch.cat([support_layer3, support_feats[3]], 1)
            mask_down = F.interpolate(support_mask.float().unsqueeze(1),
                                      size=(support_feats[3].size(2), support_feats[3].size(3)),
                                      mode='bilinear', align_corners=True)
            supp_feat = self.down_supp(supp_feat)  # torch.Size([1, 256, 60, 60])
            supp_pro = Weighted_GAP(supp_feat, mask_down)
            supp_pro_list.append(supp_pro)


            support_layer3_th = F.interpolate(support_feats_th[9],
                                              size=(support_feats_th[3].size(2), support_feats_th[3].size(3)),
                                              mode='bilinear', align_corners=True)
            supp_feat_th = torch.cat([support_layer3_th, support_feats_th[3]], 1)

            supp_feat_th = self.down_supp(supp_feat_th)  # torch.Size([1, 256, 60, 60])
            supp_pro_th = Weighted_GAP(supp_feat_th, mask_down)  # torch.Size([2, 256, 1, 1])
            t = (supp_pro - supp_pro_th) ** 2  # torch.Size([2, 256, 1, 1])
            t1 = t.view(2, -1)
            tmp = torch.sum(t1).cpu()
            tmp1 = np.sqrt(tmp)
            tmp2 = tmp1.cuda(0)
            aux_loss1 = torch.zeros_like(tmp2).cuda()
            aux_loss1 = aux_loss1 + tmp2
            ################################################################################################################################
            query_layer3 = F.interpolate(query_feats[9], size=(query_feats[3].size(2), query_feats[3].size(3)),
                                         mode='bilinear', align_corners=True)
            quy_feat = torch.cat([query_layer3, query_feats[3]], 1)
            mask_down1 = F.interpolate(history_mask.float().unsqueeze(1),
                                       size=(query_feats[3].size(2), query_feats[3].size(3)),
                                       mode='bilinear', align_corners=True)  # torch.Size([2, 1, 25, 25])
            quy_feat = self.down_query(quy_feat)  # torch.Size([1, 256, 60, 60])
            quy_pro = Weighted_GAP(quy_feat, mask_down1)

            query_layer3_th = F.interpolate(query_feats_th[9],
                                            size=(query_feats_th[3].size(2), query_feats_th[3].size(3)),
                                            mode='bilinear', align_corners=True)
            quy_feat_th = torch.cat([query_layer3_th, query_feats_th[3]], 1)

            quy_feat_th = self.down_query(quy_feat_th)  # torch.Size([1, 256, 60, 60])
            quy_pro_th = Weighted_GAP(quy_feat_th, mask_down1)  # torch.Size([2, 256, 1, 1])
            th = (quy_pro - quy_pro_th) ** 2  # torch.Size([2, 256, 1, 1])
            th1 = th.view(2, -1)
            tmp_th = torch.sum(th1).cpu()
            tmp1_th = np.sqrt(tmp_th)
            tmp2_th = tmp1_th.cuda(0)
            aux_loss2 = torch.zeros_like(tmp2_th).cuda()
            aux_loss2 = aux_loss2 + tmp2_th
#########################################################################################################################################

            #Prior Similarity Mask
            corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):
                resize_size = tmp_supp_feat.size(2)  # 60
                tmp_mask = F.interpolate(support_mask.unsqueeze(1).float(), size=(resize_size, resize_size), mode='bilinear',
                                         align_corners=True)  # torch.Size([1, 1, 60, 60])

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = query_backbone_layers[4]
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]  # 1，2048，60，60

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                corr_query = F.interpolate(corr_query, size=(
                query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                           mode='bilinear', align_corners=True)  # torch.Size([1, 1, 60, 60])
                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1)  # torch.Size([1, 1, 60, 60])


            # Support Prototype
            supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
            ###################################################################################################################################


            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
            corr_th = Correlation.multilayer_correlation(query_feats_th, support_feats_th, self.stack_ids)




        logit_mask_r = self.hpn_learner(corr)#torch.Size([1, 2, 100, 100])
        logit_mask_th = self.hpn_learner(corr_th)

        logit_mask_s =self.em4(logit_mask_th,logit_mask_r) #RAFM

        concat_feat = supp_pro.expand_as(quy)
        merge_feat = torch.cat([quy,concat_feat,corr_query_mask],1)  #MSEM
        #torch.Size([2, 256, 25, 25]),torch.Size([2, 256, 25, 25]),torch.Size([2, 1, 25, 25]),torch.Size([2, 256, 25, 25])

        merge_feat = F.interpolate(merge_feat, size=(logit_mask_s.size(2),logit_mask_s.size(3)),mode='bilinear', align_corners=True)
        merge_feat = torch.cat([merge_feat, logit_mask_s],1)#torch.Size([2, 64, 50, 50])
        merge_feat = self.init_merge(merge_feat)

        query_meta = self.res2_meta(merge_feat) + merge_feat

        logit_mask=self.decoder2(query_meta)

        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)#torch.Size([1, 2, 400, 400])

        return logit_mask,aux_loss1,aux_loss2

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)#torch.Size([1, 400, 400])-torch.Size([1, 1, 50, 50])
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)#1
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]#tensor([5])
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])#tensor([[5], [1]]),torch.Size([2, 1])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)#torch.Size([1, 1, 1])
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        if self.use_original_imgsize:
             logit_mask = F.interpolate(logit_mask, size=gt_mask.size()[1:], mode='bilinear', align_corners=True)

        logit_mask = logit_mask.view(bsz,2,-1)

        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging




