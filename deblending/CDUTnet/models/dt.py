# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1= nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, S, N, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B,  H // window_size[0] * W//window_size[1], window_size[0]*window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W,C):
    """
    Args:
        windows: (B, S, N, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] )
    x = windows.view(B, H // window_size[0], W // window_size[1],window_size[0], window_size[1],C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C).view(B,H*W,C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = (1/(3*head_dim)) ** -0.5

        #define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, S,N, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, N*C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B,S,N,C).reshape(B,S*N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x=x.reshape(B,S,N,C)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class DeblendingTransformerBlock(nn.Module):
    r""" Deblending Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=(1,128),
                 mlp_ratio=4., shift=None,qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-5)):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        if shift == 0:
            self.window_size=(window_size, input_resolution[0])
        elif shift == 1:
            self.window_size = (input_resolution[0],window_size)
        elif shift == 2:
            self.window_size = (8,8)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        shifted_x = x
            # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # B H*W//window_size**2, window_size*window_size, C,   B N S C
        x_windows = x_windows.permute(0,2,1,3).contiguous()# B window_size*window_size, H*W//window_size**2, C   B S N C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # B,window_size*window_size, H*W//window_size**2 ,C   in:B S N C   out: B S N C
        # merge windows
        attn_windows = attn_windows.permute(0,2,1,3).contiguous()  # B H*W//window_size**2, window_size*window_size, C       out:B N S C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W,C)  # B H*W C

        x = self.norm1(shifted_x)
        # FFN
        x = shortcut +x
        x = x + self.norm2(self.mlp(x))###

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

class Decoder(nn.Module):

    def __init__(self, inplanes, input_resolution,drop_block=None):
        super().__init__()
        self.input_resolution = input_resolution
        # self.de1 = nn.ConvTranspose2d(inplanes, 16, kernel_size=(3,3), stride=1, padding=1,bias=False)
        # self.deact1 =nn.LeakyReLU(0.1)######
        # self.de2= nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=1, padding=1,bias=False)
        # self.deact2 =nn.LeakyReLU(0.1)######
        self.out = nn.Conv2d(inplanes, 1, kernel_size=(3,3), stride=1, padding=1)

    def forward(self, x):
        H,W=self.input_resolution
        B,L,C=x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2).contiguous()

        # x = self.de1(x)
        # x = self.deact1(x)
        # x = self.de2(x)
        # x = self.deact2(x)
        x = self.out(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, in_chans,numbers,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-5),use_checkpoint=False):

        super().__init__()
        self.in_chans=in_chans
        self.numbers=numbers
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size=window_size
        # build blocks
        self.blocks = nn.ModuleList([
            DeblendingTransformerBlock(dim=dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, window_size=self.window_size,
                                 mlp_ratio=mlp_ratio,
                                 shift = i,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:###fine_tune
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):####
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, in_chans=64, embed_dim=64,norm_layer=None,conv_stem=True):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.conv_stem = conv_stem
        self.hid_dim=embed_dim
        self.embed_dim = embed_dim
        if conv_stem:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(self.hid_dim),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(self.hid_dim),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(self.hid_dim),
                nn.LeakyReLU(0.1))
        self.residual_conv = nn.Conv2d(in_chans, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_act=nn.LeakyReLU(0.1)
        self.proj = nn.Conv2d(self.hid_dim, self.embed_dim, kernel_size= 3, stride=1,padding=1)

    def forward(self, x):

        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        residual=x
        if self.conv_stem:
            x = self.conv(x)

        residual = self.residual_conv(residual)
        residual = self.residual_act(residual)
        x+=residual

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

class DeblendingTransformer(nn.Module):
    r""" DeblendingTransformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=128,  in_chans=1,num_classes=1,
                 embed_dim=64, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5), patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.in_chans=in_chans
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim)
        self.mlp_ratio = mlp_ratio
        self.resolution = img_size
        self.patches_resolution=to_2tuple(self.resolution)

        self.patch_embed = PatchEmbed(img_size=self.patches_resolution, in_chans=self.in_chans,
                                      embed_dim=self.num_features,norm_layer=norm_layer)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(in_chans=self.in_chans,numbers=i_layer,dim=int(self.num_features),####
                               input_resolution=(self.patches_resolution[0],
                                                 self.patches_resolution[1]),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)####norm the channels
        self.decoder = Decoder(inplanes=self.num_features,input_resolution=(self.patches_resolution[0],
                                                 self.patches_resolution[1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m.bias is None:
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.decoder(x)
        return x

