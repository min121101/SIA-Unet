import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


# class NonLocalBlock(nn.Module):
#     def __init__(self, channel):
#         super(NonLocalBlock, self).__init__()
#         self.inter_channel = channel // 2
#         self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
#         self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
#         self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#         self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
#     def forward(self, x):
#         # [N, C, H , W]
#         b, c, h, w = x.size()
#         # [N, C/2, H * W]
#         x_phi = self.conv_phi(x).view(b, c, -1)
#         # [N, H * W, C/2]
#         x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
#         x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
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

# class SelfAttention(nn.Module):
#     "Self attention layer for nd."
#     def __init__(self, n_channels:int):
#         super().__init__()
#         self.query = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
#         self.key   = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
#         self.value = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
#         self.gamma = nn.Parameter(tensor([0.]))
#
#     def forward(self, x):
#         #Notation from https://arxiv.org/pdf/1805.08318.pdf
#         size = x.size()
#         x = x.view(*size[:2],-1)
#         f,g,h = self.query(x),self.key(x),self.value(x)
#         beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
#         o = self.gamma * torch.bmm(h, beta) + x
#         return o.view(*size).contiguous()

# class self_attention(nn.Module):
#     r"""
#         Create global dependence.
#         Source paper: https://arxiv.org/abs/1805.08318
#     """
#
#     def __init__(self, in_channels):
#         super(self_attention, self).__init__()
#         self.in_channels = in_channels
#
#         self.f = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.h = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#         self.softmax_ = nn.Softmax(dim=2)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.init_weight(self.f)
#         self.init_weight(self.g)
#         self.init_weight(self.h)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#
#         assert channels == self.in_channels
#
#         f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
#         g = self.g(x).view(batch_size, -1, height * width)  # B * C//8 * (H * W)
#
#         attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
#         attention = self.softmax_(attention)
#
#         h = self.h(x).view(batch_size, channels, -1)  # B * C * (H * W)
#
#         self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W
#
#         return self.gamma * self_attention_map + x
#
#     def init_weight(self, conv):
#         nn.init.kaiming_uniform_(conv.weight)
#         if conv.bias is not None:
#             conv.bias.data.zero_()



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_DIA,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.channels = skip_channels
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.use_DIA = use_DIA
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # if (self.use_DIA):
            #     device = x.device
            #     # self.attention3 = SelfAttention(n_channels=self.channels)
            #     self.attention3 = md.Attention('scse', in_channels=self.channels)
            #     self.attention3.to(device)
            #     skip = self.attention3(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            use_DIA=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_DIA, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
