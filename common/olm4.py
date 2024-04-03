import torch
import torch.nn as nn
import torch.nn.functional as F
from common.SE import se_block
from common.PE import _PositionAttentionModule as pe_block


class olm(nn.Module):
    def __init__(self, outchannel):
        super(olm, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv5 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=5, padding=5)
        self.conv7 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=7, padding=7)

        self.conv = nn.Conv2d(5*outchannel, outchannel, 3, padding=1)
        self.convs = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3,  dilation=1,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

        # self.convf = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannel),

        )

        self.rrconv = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        self.rrbn = nn.BatchNorm2d(outchannel)
        self.rrrelu = nn.ReLU()

        self.conv0 = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)
        # self.conv0 = nn.Conv2d(outchannel, outchannel, kernel_size=1)
        self.se = ChannelAttention(outchannel)
        self.se1 = _ChannelAttentionModule()
        self.pe = SpatialAttention()
        # self.ne = NonLocalBlock(outchannel)



    def  forward(self,x, ir):
        # xx1 = x + ir
        # x_1 = self.se1(x)
        # x_2 = self.ne(x)
        # x_3 = x_1 + x_2
        # x_31 = x_3*xx1
        # ir_1=self.se1(ir)
        # ir_2=self.ne(ir)
        # ir_3=ir_1+ir_2
        # ir_31 = ir_3*xx1
        # xx = torch.cat((x_31, ir_31), dim=1)
        # xx = self.conv0(xx)
        # n = self.rconv(xx)
        # xx = self.rrrelu(xx + n)
        # x_s = self.convs(xx)


        xx1 = x + ir
        x_1 = self.se1(x)
        x_11 = x_1 * xx1
        ir_1 = self.se1(ir)
        ir_11 = ir_1 * xx1
        xx = torch.cat((x_11, ir_11), dim=1)
        xx = self.conv0(xx)
        n = self.rconv(xx)
        xx = self.rrrelu(xx + n)
        x_s = self.convs(xx)


        return x_s



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out

class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



# class NonLocalBlock(nn.Module):
#     def __init__(self, channel):
#         super(NonLocalBlock, self).__init__()
#         self.inter_channel = channel // 2
#         self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
#         self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
#         self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#         self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
#
#     def forward(self, x):
#         # [N, C, H , W]
#         b, c, h, w = x.size()
#         # [N, C/2, H * W]
#         x_phi = self.conv_phi(x).view(b, c//2, -1)
#         # [N, H * W, C/2]
#         x_theta = self.conv_theta(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
#         x_g = self.conv_g(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
#         # [N, H * W, H * W]
#         mul_theta_phi = torch.matmul(x_theta, x_phi)
#         mul_theta_phi = self.softmax(mul_theta_phi)
#         # [N, H * W, C/2]
#         mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
#         # [N, C/2, H, W]
#         mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
#         # [N, C, H , W]
#         mask = self.conv_mask(mul_theta_phi_g)
#         out = mask + x
#         return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
