r""" Implementation of center-pivot 4D convolution """

import torch
import torch.nn as nn


class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size() #1,3,13,13,13,13
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)#tensor([ 0,  2,  4,  6,  8, 10, 12])
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)#tensor([ 0,  2,  4,  6,  8, 10, 12])
            self.len_h = len(idxh)#7
            self.len_w = len(idxw)#7
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)
        #torch.Size([1, 3, 13, 13, 7, 7])
        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()#torch.Size([1, 3, 13, 13, 7, 7])
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)#torch.Size([49, 3, 13, 13])
        out1 = self.conv1(out1)#torch.Size([49, 16, 13, 13])
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)#16,13,13
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()#torch.Size([1, 3, 13, 13, 13, 13])
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)#torch.Size([169, 3, 13, 13])
        out2 = self.conv2(out2)#torch.Size([169, 16, 7, 7])
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()#torch.Size([1, 16, 13, 13, 7, 7])

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)#torch.Size([1, 16, 7, 7, 4, 4])
            out2 = out2.squeeze()#torch.Size([1, 16, 7, 7, 4, 4])

        y = out1 + out2 #torch.Size([1, 16, 13, 13, 7, 7])
        return y
