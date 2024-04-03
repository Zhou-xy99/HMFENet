r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from common.olm4 import olm
import numpy as np

from .base.swin_transformer import SwinTransformer
from model.base.transformer import MultiHeadedAttention, PositionalEncoding


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class DCAMA(nn.Module):

    def __init__(self, backbone, use_original_imgsize):
        super(DCAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load("backbones\\swin_base_patch4_window12_384.pth"))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load("backbones\\swin_base_patch4_window12_384.pth"))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load("backbones\\swin_base_patch4_window12_384.pth")['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)
        # self.model_th = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)

        self.em4 = olm(2)
        reduce_dim = 8
        fea_dim = 256
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, query_img_th, support_img, support_img_th, support_mask, history_mask):
    # def forward(self, query_img, support_img, support_mask):

        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)
            query_feats_th = self.extract_feats(query_img_th)
            support_feats_th = self.extract_feats(support_img_th)

        supp_feat = F.interpolate(support_feats[3], size=(support_feats[3].size(2), support_feats[3].size(3)),
                                       mode='bilinear', align_corners=True)
        # supp_feat = torch.cat([support_layer3, support_feats[3]], 1)
        mask_down = F.interpolate(support_mask.float().unsqueeze(1),
                                  size=(support_feats[3].size(2), support_feats[3].size(3)),
                                  mode='bilinear', align_corners=True)
        supp_feat = self.down_supp(supp_feat)  # torch.Size([1, 256, 60, 60])
        supp_pro = Weighted_GAP(supp_feat, mask_down)

        supp_feat_th = F.interpolate(support_feats_th[3],
                                          size=(support_feats_th[3].size(2), support_feats_th[3].size(3)),
                                          mode='bilinear', align_corners=True)
        # supp_feat_th = torch.cat([support_layer3_th, support_feats_th[3]], 1)

        supp_feat_th = self.down_supp(supp_feat_th)  # torch.Size([1, 256, 60, 60])
        supp_pro_th = Weighted_GAP(supp_feat_th, mask_down)  # torch.Size([2, 256, 1, 1])

        #############################################################################################
        x = supp_pro.view(-1)
        y = supp_pro_th.view(-1)
        aux_loss1 = self.mutual_information(x.tolist(), y.tolist())
        #############################################################################################

        # t = (supp_pro - supp_pro_th) ** 2  # torch.Size([2, 256, 1, 1])
        # t1 = t.view(2, -1)
        # tmp = torch.sum(t1).cpu().detach().numpy()
        # tmp = torch.tensor(tmp)
        # tmp2 = tmp.cuda()
        # aux_loss1 = torch.zeros_like(tmp2).cuda()
        # aux_loss1 = aux_loss1 + tmp2
        ################################################################################################################################
        quy_feat = F.interpolate(query_feats[3], size=(query_feats[3].size(2), query_feats[3].size(3)),
                                     mode='bilinear', align_corners=True)
        # quy_feat = torch.cat([query_layer3, query_feats[3]], 1)
        mask_down1 = F.interpolate(history_mask.float().unsqueeze(1),
                                   size=(query_feats[3].size(2), query_feats[3].size(3)),
                                   mode='bilinear', align_corners=True)  # torch.Size([2, 1, 25, 25])
        quy_feat = self.down_query(quy_feat)  # torch.Size([1, 256, 60, 60])
        quy_pro = Weighted_GAP(quy_feat, mask_down1)

        quy_feat_th = F.interpolate(query_feats_th[3],
                                        size=(query_feats_th[3].size(2), query_feats_th[3].size(3)),
                                        mode='bilinear', align_corners=True)
        # quy_feat_th = torch.cat([query_layer3_th, query_feats_th[3]], 1)

        quy_feat_th = self.down_query(quy_feat_th)  # torch.Size([1, 256, 60, 60])
        quy_pro_th = Weighted_GAP(quy_feat_th, mask_down1)  # torch.Size([2, 256, 1, 1])

        #############################################################################################
        x_q = quy_pro.view(-1)
        y_q = quy_pro_th.view(-1)
        aux_loss2 = self.mutual_information(x_q.tolist(), y_q.tolist())
        #############################################################################################

        # th = (quy_pro - quy_pro_th) ** 2  # torch.Size([2, 256, 1, 1])
        # th1 = th.view(2, -1)
        #
        # tmp_th = torch.sum(th1).cpu().detach().numpy()
        # tmp_th = torch.tensor(tmp_th)
        # tmp2_th = tmp_th.cuda()
        # aux_loss2 = torch.zeros_like(tmp2_th).cuda()
        # aux_loss2 = aux_loss2 + tmp2_th

        #############################################################################################

        logit_mask = self.model(query_feats, support_feats, support_mask.clone())
        logit_mask_th = self.model(query_feats_th, support_feats_th, support_mask.clone())

        # logit_mask,query_8, query_16, query_32 = self.model(query_feats, support_feats, support_mask.clone())
        # logit_mask_th,_,_,_ = self.model(query_feats_th, support_feats_th, support_mask.clone())

        logit_mask_s = self.em4(logit_mask_th, logit_mask)

        #############################################################################################
        #
        # # mask_th_pro = Weighted_GAP(logit_mask_th, logit_mask_s)
        # # mask_pro = Weighted_GAP(logit_mask, logit_mask_s)
        # th_th = (logit_mask_th - logit_mask_s) ** 2
        #
        # th = (logit_mask - logit_mask_s) ** 2
        # th1 = th.view(2, -1)
        #
        # th_th1= th_th.view(2, -1)
        #
        # tmp_th = torch.sum(th1).cpu().detach().numpy() + torch.sum(th_th1).cpu().detach().numpy()
        # tmp_th = torch.tensor(tmp_th / th1.numel())
        # tmp2_th = tmp_th.cuda()
        # pro_loss = torch.zeros_like(tmp2_th).cuda()
        # pro_loss = pro_loss + tmp2_th
        #
        # # print(pro_loss)

        #############################################################################################
        # return logit_mask

        # aux_loss = aux_loss1 + aux_loss2
        return logit_mask_s, aux_loss1, aux_loss2
        # return logit_mask_s, aux_loss1, aux_loss2, query_8, query_16, query_32

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, support_imgs[:, 0], support_masks[:, 0])
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()

    def mutual_information(self, feature1, feature2, bins=10):
        # 将特征转换为Tensor
        feature1 = torch.Tensor(feature1)
        feature2 = torch.Tensor(feature2)

        # 计算特征的直方图
        hist_1 = torch.histc(feature1, bins=bins, min=feature1.min(), max=feature1.max())
        hist_2 = torch.histc(feature2, bins=bins, min=feature2.min(), max=feature2.max())

        # 计算两个特征的联合直方图
        hist_12, _, _ = np.histogram2d(feature1, feature2, bins=bins)
        hist_12 = torch.tensor(hist_12)

        # 添加拉普拉斯平滑
        hist_1 += 1
        hist_2 += 1
        hist_12 += 1

        # 计算概率分布
        p_1 = hist_1 / feature1.numel()
        p_2 = hist_2 / feature2.numel()
        p_12 = hist_12 / feature1.numel()

        # 计算互信息
        mi = torch.sum(p_12 * torch.log2(p_12 / (p_1.unsqueeze(1) * p_2)))

        return mi.item()



class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks outch3+2*in_channels[1]+2*in_channels[0]
        #nn.Sequential(nn.Conv2d(outch3+2*in_channels[3]
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

        # self.mixer1 = nn.Sequential(nn.Conv2d(outch3 + 2 * in_channels[1], outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())
        #
        # self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())
        #
        # self.mixer3 = nn.Sequential(nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        coarse_masks = []

        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()

            # if idx == 3:
            #     query_8 = query
            # elif idx == 21:
            #     query_16 = query
            # elif idx == 23:
            #     query_32 = query

            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)
            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        # mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        # mix = self.conv5(mix)
        # upsample_size = (mix.size(-1) * 2,) * 2
        # mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)


        # mix = coarse_masks1
        # mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        #
        #
        # ################ skip connect 1/8 and 1/4 features (concatenation) ##############
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]#3
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        ################################################################################
        # if nshot == 1:
        #     support_feat = support_feats[self.stack_ids[3] - 1]#3
        # else:
        #     support_feat = torch.stack([support_feats[k][self.stack_ids[3] - 1] for k in range(nshot)]).max(dim=0).values
        # upsample_size_0 = (support_feat.size(-1) * 4,) * 2
        # support_feat = F.interpolate(support_feat, upsample_size_0, mode='bilinear', align_corners=True)
        #
        # query_feat = query_feats[self.stack_ids[3] - 1]
        # upsample_size_1 = (query_feat.size(-1) * 4,) * 2
        # query_feat = F.interpolate(query_feat, upsample_size_1, mode='bilinear', align_corners=True)
        #
        # mix = torch.cat((mix, query_feat, support_feat), 1)
        #
        # #################################################################
        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        # ################################################################################

        # if nshot == 1:
        #     support_feat = support_feats[self.stack_ids[0] - 1] #1
        # else:
        #     support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        # mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)
        ################################################################################

        # mixer blocks forward
        out = self.mixer1(mix)
        # print(out.size())
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        # print(out.size())
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask
        # return logit_mask, query_8, query_16, query_32

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
