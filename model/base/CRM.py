
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(d_model,max_len)#torch.Size([10000, 512])
        position = torch.arange(0, max_len)#torch.Size([10000, 1])
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)).unsqueeze(1)#256
        pe[0::2,:] = torch.sin(div_term*position)
        pe[1::2,:] = torch.cos(div_term*position)
        pe = pe.unsqueeze(0)#torch.Size([1,512,10000])
        self.register_buffer('pe', pe)#模型的常数,是一个持久态，不会有梯度传播给它，但是能被模型的state_dict记录下来

    def forward(self, x):
        x = x + Variable(self.pe[:,:, :x.size(2)],
                         requires_grad=False)
        return self.dropout(x)

class GCCG(torch.nn.Module):
    def __init__(self, context_size=5, output_channel=128):
        super(GCCG, self).__init__()
        self.context_size = context_size
        self.pad = context_size // 2
        self.conv = torch.nn.Conv2d(
            self.context_size * self.context_size,
            output_channel * 2,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        self.conv1 = torch.nn.Conv2d(
            output_channel * 2,
            output_channel,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        # additional layer
        # self.conv2 = torch.nn.Conv2d(
        #     output_channel,
        #     output_channel,
        #     3,
        #     padding=(1, 1),
        #     bias=True,
        #     padding_mode="zeros",
        # )

    def self_similarity(self, feature_normalized):
        b, c, h, w = feature_normalized.size()
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [self.context_size * self.context_size, b, h, w],
            dtype=feature_normalized.dtype,
            requires_grad=feature_normalized.requires_grad,
        )
        if feature_normalized.is_cuda:
            output = output.cuda(feature_normalized.get_device())

        # with torch.no_grad():
        for c in range(self.context_size):
            for r in range(self.context_size):
                i = c * self.context_size + r
                a=output.clone()
                a[i] = (
                        feature_pad[:, :, r: (h + r), c: (w + c)] * feature_normalized
                ).sum(1)


        output = output.transpose(0, 1).contiguous()
        return output

    def forward(self, feature):
        feature_normalized = F.normalize(feature, p=2, dim=1)
        ss = self.self_similarity(feature_normalized)

        ss1 = F.relu(self.conv(ss))
        ss2 = F.relu(self.conv1(ss1))
        output = torch.cat((ss, ss1, ss2), 1)
        return output

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,heads=8, groups=4):
        super(Attention, self).__init__()
        self.heads = heads
        '''
        Size of conv output = floor((input  + 2 * pad - kernel) / stride) + 1
        The second condition of `retain_dim` checks the spatial size consistency by setting input=output=0;
        Use this term with caution to check the size consistency for generic cases!
        '''
        retain_dim = in_channels == out_channels
        hidden_channels = out_channels // 2
        assert hidden_channels % self.heads == 0, "out_channels should be divided by heads. (example: out_channels: 40, heads: 4)"

        ksz_q = (1, 1)
        str_q = (1, 1)
        pad_q = (0, 0)

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        ) if not retain_dim else nn.Identity()#输入是啥，直接给输出，不做任何的改变

        # Convolutional embeddings for (q, k, v)
        self.qhead = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=bias)

        ksz = (1, 1)
        str = (1, 1)
        pad = (0, 0)

        self.khead = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)
        self.vhead = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)

        self.agg = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)
        self.pe = PositionalEncoding(d_model=in_channels // 2, dropout=0.5)

    def forward(self, input):
        x = input#torch.Size([2, 3, 169, 13, 13])，torch.Size([2, 400, 400])

        x_ = self.short_cut(x)#torch.Size([2, 3, 169, 13, 13])
        q_out = self.qhead(x)#torch.Size([2, 16, 169, 4, 4]) b c (d t) h w'
        k_out = self.khead(x)#torch.Size([2, 16, 169, 13, 13])
        v_out = self.vhead(x)#torch.Size([2, 16, 169, 13, 13])

        q_h, q_w = q_out.shape[-2:]  # 4，4
        k_h, k_w = k_out.shape[-2:]  # 13，13

        q_out = rearrange(q_out, 'b  c  h w -> b c (h w)')#torch.Size([2, 8, 2, 169, 16])
        k_out = rearrange(k_out, 'b  c  h w -> b c (h w)')#torch.Size([2, 8, 2, 169, 169])
        v_out = rearrange(v_out, 'b  c  h w -> b c (h w)')#torch.Size([2, 8, 2, 169, 169])
        q_out,k_out,v_out=self.pe(q_out), self.pe(k_out),self.pe(v_out)

        out = torch.einsum('b c l, b c m -> b l m', q_out, k_out)#torch.Size([2, 8, 169, 16, 169])
        out = F.softmax(out, dim=-1)#torch.Size([2, 8, 169, 16, 169])
        out = torch.einsum('b l m, b c m -> b c l', out, v_out)#torch.Size([2, 8, 2, 169, 16])
        out = rearrange(out, 'b c (h w) -> b c h w', h=q_h, w=q_w)#torch.Size([2, 16, 169, 4, 4])
        out = self.agg(out)#torch.Size([2, 32, 169, 4, 4])

        return self.out_norm(out + x_)

class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3,1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1,3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, x):
        # feature_size = x.shape[-1]#torch.Size([2, 512, 25, 25])
        # ch1 = F.avg_pool2d(x, kernel_size=(1, feature_size))#torch.Size([2, 512, 25, 1])
        # output1 = self.conv3x1_1(ch1)#torch.Size([2, 512, 25, 1])
        # output1 = self.act(output1)
        # ch2 = F.avg_pool2d(x, kernel_size=(feature_size,1))#torch.Size([2, 512, 1, 25])
        # output2 = self.conv1x3_1(ch2)#torch.Size([2, 512, 1, 25])
        # # output2 = self.bn1(output2)
        # output2 = self.act(output2)
        # output=output1+output2#torch.Size([2, 512, 25, 25])

        feature_size = x.shape[-1]  # torch.Size([2, 512, 25, 25])
        ch1 = F.avg_pool2d(x, kernel_size=(1, feature_size))  # torch.Size([2, 512, 25, 1])
        output1 = self.conv3x1_1(ch1)  # torch.Size([2, 512, 25, 1])
        # output1 = self.act(output1)
        output1 = F.interpolate(output1,
                                size=(x.size(2), x.size(3)),
                                mode='bilinear', align_corners=True)
        para1 = output1.softmax(-2)
        ch2 = F.avg_pool2d(x, kernel_size=(feature_size, 1))  # torch.Size([2, 512, 1, 25])
        output2 = self.conv1x3_1(ch2)  # torch.Size([2, 512, 1, 25])
        # output2 = self.bn1(output2)
        # output2 = self.act(output2)
        output2 = F.interpolate(output2,
                                size=(x.size(2), x.size(3)),
                                mode='bilinear', align_corners=True)
        para2 = output2.softmax(-1)

        output = para1 * output1 + para2 * output2  # torch.Size([2, 512, 25, 25])



        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + x)