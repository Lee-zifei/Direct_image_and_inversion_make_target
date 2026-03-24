import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from .ham import NMF2D
import numpy as np
class OverlapPatchEmbed(nn.Module):####

    def __init__(self, img_size=128, in_chans=1, embed_dim=32,res_stem=False):
        super(OverlapPatchEmbed,self).__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.hid_dim=embed_dim
        self.embed_dim = embed_dim
        self.res_stem = res_stem
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True))
        if res_stem:
            self.residual_conv = nn.Conv2d(in_chans, self.hid_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_act = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.res_stem:
            residual = x
            x = self.conv(x)
            residual = self.residual_conv(residual)
            residual = self.residual_act(residual)
            x+=residual
        else:
            x = self.conv(x)
        return x

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,L,C = x.shape
        x = x.reshape(B, int(L**0.5), int(L**0.5), C).permute(0,3,1,2).contiguous()
        x = self.fc1(x)
        x = self.act(self.dwconv(x))
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Ham(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()
        C = getattr(args, 'MD_D', 256)
        self.lower_bread = nn.Sequential(nn.Conv2d(in_c, C, 1),
                                             nn.ReLU(inplace=True))
        self.ham = NMF2D(args)
        self.cheese = nn.Sequential(nn.Conv2d(C,C,1),nn.BatchNorm2d(C),nn.ReLU(inplace=True))
        self.upper_bread = nn.Conv2d(C, in_c, 1, bias=False)
        self.shortcut = nn.Sequential()
        self._init_weight()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.cheese(x)
        x = self.upper_bread(x)
        x = x + shortcut
        x = F.relu(x, inplace=True)
        return x

class NMFfomer(nn.Module):

    def __init__(self, dim, input_resolution, depth, mlp_ratio=2., drop=0.,
                 drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-5)):
        super(NMFfomer,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList([
            NMFBlock(dim=dim, input_resolution=self.input_resolution,mlp_ratio=mlp_ratio,
                     drop=drop, drop_path=drop_path,norm_layer=norm_layer)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim,dim,3,1,1)
        self.act = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.conv(x))
        return x

class NMFBlock(nn.Module):

    def __init__(self, dim,  input_resolution, mlp_ratio=4., drop=0.1, drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-5)):
        super(NMFBlock,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.attn = Ham(dim)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.flatten(2).transpose(1, 2))))
        return x

class ReduceChannel(nn.Module):
    def __init__(self,dim,rate=2):
        super(ReduceChannel,self).__init__()
        self.rate = rate
        self.conv = nn.Conv2d(dim,dim//self.rate,kernel_size=1,padding=0,bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):

    def __init__(self, inplanes):
        super(Decoder,self).__init__()
        self.out = nn.Conv2d(inplanes, 1, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.out(x)
        return x

class NMFnet(nn.Module):

    def __init__(self, img_size=128,  in_chans=1,num_classes=1,
                 embed_dim=64, enc_depths=[2, 4, 4], dec_depths=[2,2],mlp_ratio=2.,
                 drop_rate=0., drop_path_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-5),
                 patch_norm=True,config=None, **kwargs):

        super(NMFnet,self).__init__()
        self.in_chans=in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim)
        self.mlp_ratio = mlp_ratio
        self.resolution = img_size
        self.patches_resolution=to_2tuple(self.resolution)

        self.patch_embed = OverlapPatchEmbed(img_size=self.patches_resolution, in_chans=self.in_chans,
                                      embed_dim=self.num_features)
        # build layers
        self.decorder1=nn.Sequential(
                                NMFfomer(dim=int(self.num_features),input_resolution=(self.patches_resolution[0],self.patches_resolution[1]),depth=enc_depths[0],
                                mlp_ratio=self.mlp_ratio,drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer))

        self.down1=nn.Sequential(nn.Conv2d(self.num_features, self.num_features//2, kernel_size=3, stride=1, padding=1, bias=False),DWT())

        self.decorder2=nn.Sequential(
                                NMFfomer(dim=int(self.num_features*2**1),input_resolution=(self.patches_resolution[0]//2,self.patches_resolution[1]//2),depth=enc_depths[1],
                                mlp_ratio=self.mlp_ratio,drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer))

        self.down2=nn.Sequential(nn.Conv2d(self.num_features*2**1, self.num_features, kernel_size=3, stride=1, padding=1, bias=False),DWT())
        self.center=nn.Sequential(
                                NMFfomer(dim=int(self.num_features*2**2),input_resolution=(self.patches_resolution[0]//4,self.patches_resolution[1]//4),depth=enc_depths[2],
                               mlp_ratio=self.mlp_ratio,drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer))

        self.up2=nn.Sequential(nn.Conv2d(self.num_features*2**2, self.num_features*2*2**2, kernel_size=3, stride=1, padding=1, bias=False),IWT())

        self.reduce_channel2=ReduceChannel(dim=int(self.num_features*2*2**1))

        self.encoder2=nn.Sequential(
                                NMFfomer(dim=int(self.num_features*2**1),input_resolution=(self.patches_resolution[0]//2,self.patches_resolution[1]//2),depth=dec_depths[1],
                               mlp_ratio=self.mlp_ratio,drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer))

        self.up1 = nn.Sequential(nn.Conv2d(self.num_features*2**1, self.num_features*2*2**1, kernel_size=3, stride=1, padding=1, bias=False),IWT())

        self.reduce_channel1=ReduceChannel(dim=int(self.num_features*2))

        self.encoder1=nn.Sequential(
                                NMFfomer(dim=int(self.num_features),input_resolution=(self.patches_resolution[0],self.patches_resolution[1]),depth=dec_depths[0],
                               mlp_ratio=self.mlp_ratio,drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer))

        self.decoder = Decoder(inplanes=int(self.num_features))

    def forward_features(self, x):

        dec1 = self.decorder1(x)
        ds1 = self.down1(dec1)

        dec2 = self.decorder2(ds1)
        ds2 = self.down2(dec2)

        cen = self.center(ds2)

        us2 = self.up2(cen)
        cat2 = torch.cat([dec2, us2], 1)
        reduce2 = self.reduce_channel2(cat2)
        enc2 = self.encoder2(reduce2)

        us1 = self.up1(enc2)
        cat1 = torch.cat([dec1, us1], 1)
        reduce1 = self.reduce_channel1(cat1)
        enc1 = self.encoder1(reduce1)
        return enc1

    def forward(self, x1):
        shortcut=x1
        x1 = self.patch_embed(x1)
        x1 = self.forward_features(x1)
        x1 = self.decoder(x1)
        return x1 + shortcut
