r""" Provides functions that builds/manipulates correlation tensors """
import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()#1，512，50，50
            support_feat = support_feat.view(bsz, ch, -1)#torch.Size([1, 512, 2500])
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)#torch.Size([1, 512, 2500])

            bsz, ch, ha, wa = query_feat.size()#torch.Size([1, 512, 50, 50])
            query_feat = query_feat.view(bsz, ch, -1)#torch.Size([1, 512, 2500])
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)#torch.Size([1, 512, 2500])

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)#torch.Size([1, 50, 50, 50, 50])
            corr = corr.clamp(min=0)#将给定的张量的所有元素的取值限定在一个指定的范围之内，都变成正的torch.Size([1, 50, 50, 50, 50])
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()#torch.Size([1, 3, 13, 13, 13, 13])
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()#torch.Size([1, 6, 25, 25, 25, 25])
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()#torch.Size([1, 4, 50, 50, 50, 50])

        return [corr_l4, corr_l3, corr_l2]

    @classmethod
    def multilayer_correlation_msa(cls, query_feats, support_feats, stack_ids):  # tensor([ 3,  9, 13])
        eps = 1e-5
        corrs = []
        sups = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape  # b,c,h,w torch.Size([1, 256, 119, 119])
            #torch.Size([2, 512, 25, 25])
            corrI = []
            realSupI = []
            for j in range(len(support_feat)):  # b
                queryIJ = query_feat[j].flatten(start_dim=1)  # c,hw torch.Size([256, 14161])/torch.Size([512, 625])
                queryIJNorm = queryIJ / (queryIJ.norm(dim=0, p=2, keepdim=True) + eps)  # torch.Size([256, 14161])
                supIJ = support_feat[j].flatten(start_dim=1) # c,hw torch.Size([256, 1099])
                supIJNorm = supIJ / (supIJ.norm(dim=0, p=2, keepdim=True) + eps)  # torch.Size([256, 1099])
                corr = (queryIJNorm.permute(1, 0)).matmul(supIJNorm)  # torch.Size([14161, 1099])
                corr = corr.clamp(min=0)
                corr = corr.mean(dim=1, keepdim=True)  # 在针对于supp的维度求均值torch.Size([14161, 1])
                corr = (corr.permute(1, 0)).unsqueeze(0)  # 1,1,hw  torch.Size([1, 1, 14161])
                corrI.append(corr)  # b,1,hw
                # resupJ=supIJ.mean(dim=1,keepdim=True)
                # resupJsum=resupJ.sum()
                # resupJ=resupJ.unsqueeze(0).expand(-1,-1,queryIJ.shape[-1])#1,c,hw
                # queryIJ=queryIJ.unsqueeze(0)#1,c,hw
                # if resupJsum==0:
                #     queryIJ=queryIJ*resupJ
                # resupJ=torch.cat([queryIJ,resupJ],dim=1)#1,2c,hw
                # realSupI.append(resupJ)#b,2c,hw
            corrI = torch.cat(corrI, dim=0)  # b,1,h,w
            corrI = corrI.reshape((corrI.shape[0], corrI.shape[1], queryShape[-2], queryShape[-1]))  # b,1,h,w
            # realSupI=torch.cat(realSupI,dim=0)#b,2c,h,w
            # realSupI=realSupI.reshape((realSupI.shape[0],realSupI.shape[1],queryShape[-2],queryShape[-1]))
            corrs.append(corrI)  # n,b,1,h,w
            # sups.append(realSupI)#n,b,c,h,w

        corr_l4 = torch.cat(corrs[-stack_ids[0]:], dim=1).contiguous()  # b,n,h,w torch.Size([1, 3, 60, 60])
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]], dim=1).contiguous()  # torch.Size([1, 6, 60, 60])
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]], dim=1).contiguous()  # torch.Size([1, 4, 60, 60])

        # sup_l4=sups[-stack_ids[0]:]#n,b,2c,h,w
        # sup_l3=sups[-stack_ids[1]:-stack_ids[0]]
        # sup_l2=sups[-stack_ids[2]:-stack_ids[1]]
        # print(corr_l4.shape,corr_l3.shape,corr_l2.shape)#n,b,1,h,wtorch.Size([13, 3, 15, 15])
        # print(len(sup_l4), len(sup_l3), len(sup_l2))
        return [corr_l4, corr_l3, corr_l2]  # ,[sup_l4,sup_l3,sup_l2]

