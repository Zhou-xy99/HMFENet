import torch.nn as nn
import torch.nn.functional as F

from .base.conv4d import CenterPivotConv4d as Conv4d


class HPNLearner(nn.Module):
    def __init__(self, inch):#inch=3,6,4
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1] #3,
                ksz4d = (ksz,) * 4 #(3, 3, 3, 3)
                str4d = (1, 1) + (stride,) * 2 #(1, 1, 2, 2)
                pad4d = (ksz // 2,) * 4#(1, 1, 1, 1)

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16*2, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()#1,128,13,13,2,2
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)#torch.Size([4, 128, 13, 13])
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)#torch.Size([4, 128, 25, 25])
        o_hb, o_wb = spatial_size#25,25
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])#3
        #torch.Size([1, 3, 13, 13, 13, 13])-torch.Size([1, 128, 13, 13, 2, 2])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])#6
        #torch.Size([1, 6, 25, 25, 25, 25])-torch.Size([1, 128, 25, 25, 2, 2])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])#4
        #torch.Size([1, 4, 50, 50, 50, 50])-torch.Size([1, 128, 50, 50, 2, 2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])#torch.Size([1, 128, 25, 25, 2, 2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3#torch.Size([1, 128, 25, 25, 2, 2])
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)#torch.Size([1, 128, 25, 25, 2, 2])

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])#torch.Size([1, 128, 50, 50, 2, 2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2#torch.Size([1, 128, 50, 50, 2, 2])
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)#torch.Size([1, 128, 50, 50, 2, 2])

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)#torch.Size([1, 128, 50, 50])

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)#torch.Size([1, 64, 50, 50])
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2#(100, 100)
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)#torch.Size([1, 64, 100, 100])
        # logit_mask = self.decoder2(hypercorr_decoded)#torch.Size([1, 2, 100, 100])在通道数变为2出结果之前合并RGB和T

        return hypercorr_decoded
