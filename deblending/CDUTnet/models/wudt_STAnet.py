import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from einops import rearrange
from seislet.inversion import Inversion
from .NAFNet_arch import  NAFNet
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
    out_batch, out_channel, out_height, out_width = in_batch, in_channel // (r**2),\
        r * in_height, r * in_width
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

class ReduceChannel(nn.Module):
    def __init__(self,dim,rate=2):
        super(ReduceChannel,self).__init__()
        self.rate = rate
        self.conv = nn.Conv2d(dim,dim//self.rate,kernel_size=1,padding=0,bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

################# Super Token #################################
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        
    def forward(self, x):
        return x + self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x


class KAN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 stoken_size, 
                 n_iter=1, 
                 refine=True, 
                 refine_attention=True, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        # 976
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        # 955
        
        # 935
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        # 911
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                # 874
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                # 871
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                    # 777
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    # 853
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    # 777            
                    
                    # 771
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                    # 767
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        # 853
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        # 767
        
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.reshape(B, C, hh*ww).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, hh, ww)
            else:
                stoken_features = self.stoken_refine(stoken_features)
            
        # 727
        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        # 714
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        # 687
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class StokenAttentionLayer(nn.Module):
    def __init__(self, 
                 dim, 
                 n_iter, 
                 stoken_size, 
                 num_heads=1, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 layerscale=False, 
                 init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x

class BasicLayerDT2(nn.Module):        
    def __init__(self, num_layers, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5,):
        super().__init__()        
                
        self.blocks = nn.ModuleList([StokenAttentionLayer(
                                        dim=dim,  
                                        n_iter=n_iter, stoken_size=stoken_size,                   num_heads=num_heads, 
                                        mlp_ratio=mlp_ratio, 
                                        qkv_bias=qkv_bias, 
                                        qk_scale=qk_scale, 
                                        drop=drop,
                                        attn_drop=attn_drop, 
                                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                        act_layer=act_layer, 
                                        layerscale=layerscale, init_values=init_values) for i in range(num_layers)])
         
    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

################# Swin trans #################################

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
    B = windows.shape[0]
    x = windows.view(B, H // window_size[0], W // window_size[1],window_size[0], window_size[1],C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C).view(B,H*W,C)
    return x

class Mlp_nores(nn.Module):
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

    def forward(self, x, H):
        B,L,C = x.shape
        x = x.reshape(B, H, H, C).permute(0,3,1,2).contiguous()
        x = self.fc1(x)
        x = self.act(self.dwconv(x))
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0,2,3,1).contiguous().reshape(B, -1, C)
        return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention,self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m.bias is not None:
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, S,N, C = x.shape
        x = x.view(B, S * N, C)
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, N*C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1))* self.scale

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

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=1,
                 mlp_ratio=4., shift=None,qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-5)):
        super(SwinTransformerBlock,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        if shift == 0 or shift==3:
            self.window_size = (input_resolution[0],window_size)
        elif shift == 1 or shift==4:
            self.window_size=(window_size, input_resolution[0])
        elif shift == 2 or shift==5:
            self.window_size=(8, 8)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_nores(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H,W=self.input_resolution
        B,L,C=x.shape
        shortcut = x      # B H*W C
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # B H*W//window_size**2, window_size*window_size, C,   B N S C
        x_windows = x_windows.permute(0, 2, 1,3).contiguous()  # B window_size*window_size, H*W//window_size**2, C   B S N C
        attn_windows = self.attn(x_windows)  # B,window_size*window_size, H*W//window_size**2 ,C   in:B S N C   out: B S N  C
        attn_windows = attn_windows.permute(0, 2, 1,3).contiguous()  # B H*W//window_size**2, window_size*window_size, C       out:B N S C
        x = window_reverse(attn_windows, self.window_size, H, W,C)  # B H*W C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x),H))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

class BasicLayerDT1(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-5)):

        super(BasicLayerDT1,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size=window_size
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, window_size=self.window_size,
                                 mlp_ratio=mlp_ratio,shift = i,qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,drop_path=drop_path,norm_layer=norm_layer)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim,dim,3,1,1,bias=False)
        self.act = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        B,L,C = x.shape
        H,W = self.input_resolution
        x = x.view(B,H,W,C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class Decoder(nn.Module):

    def __init__(self, inplanes):
        super(Decoder,self).__init__()
        self.out = nn.Conv2d(inplanes, 1, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.out(x)
        return x

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
            self.conv = nn.Sequential(
                    nn.Conv2d(in_chans, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(0.1))
            self.residual_conv = nn.Conv2d(in_chans, self.hid_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.residual_act = nn.LeakyReLU(0.1)
        self.proj = nn.Conv2d(self.hid_dim, self.embed_dim, kernel_size= 1, stride=1,padding=0)

    def forward(self, x):
        if self.res_stem:
            residual = x
            x = self.conv(x)
            residual = self.residual_conv(residual)
            residual = self.residual_act(residual)
            x+=residual
        else:
            x = self.conv(x)
        x = self.proj(x)
        return x

class WUDT_STAnet(nn.Module):

    def __init__(self, img_size=128,  in_chans=1,num_classes=1,
                embed_dim=64, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                n_iter=[1, 1, 1, 1],stoken_size=[8, 4, 2, 1],
                mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                freeze_bn=False,
                window_size=[1,1,1],norm_layer=partial(nn.LayerNorm, eps=1e-5),
                layerscale=[False, False, False, False], init_values=1e-6,config=None, **kwargs):

        super(WUDT_STAnet,self).__init__()
        self.depths=depths
        self.in_chans=in_chans
        
        self.num_features = int(embed_dim)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.n_iter = n_iter
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.freeze_bn = freeze_bn
        
        self.resolution = img_size
        self.patches_resolution=to_2tuple(self.resolution)
        self.config=config
        self.patch_embed = OverlapPatchEmbed(img_size=self.patches_resolution, in_chans=self.in_chans,
                                      embed_dim=self.num_features)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # build layers
        i_layer = 0
        self.decorder1=nn.Sequential(
                                BasicLayerDT2(
                                    num_layers=depths[i_layer],
                                    dim=int(self.num_features),
                                    n_iter=n_iter[i_layer],
                                    stoken_size=to_2tuple(stoken_size[i_layer]),   
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],       
                                    layerscale=layerscale[i_layer],
                                    init_values=init_values))

        self.down1=nn.Sequential(nn.Conv2d(self.num_features, self.num_features//2, kernel_size=3, stride=1, padding=1, bias=False),DWT())
        # 64-128
        i_layer = i_layer+1 #128
        self.decorder2=nn.Sequential(
                                BasicLayerDT2(
                                    num_layers=depths[i_layer],
                                    dim=int(self.num_features*2**1),
                                    n_iter=n_iter[i_layer],
                                    stoken_size=to_2tuple(stoken_size[i_layer]),   
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],       
                                    layerscale=layerscale[i_layer],
                                    init_values=init_values))

        self.down2=nn.Sequential(nn.Conv2d(self.num_features*2**1, self.num_features, kernel_size=3, stride=1, padding=1, bias=False),DWT())
        # 128-320
        i_layer = i_layer+1 #320
        self.center=nn.Sequential(
                                BasicLayerDT1(
                                    dim=int(self.num_features*2**2),input_resolution=(self.patches_resolution[0]//4,self.patches_resolution[1]//4),depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],window_size=window_size[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate,norm_layer=norm_layer))

        self.up2=nn.Sequential(nn.Conv2d(self.num_features*2**2, self.num_features*2*2**2, kernel_size=3, stride=1, padding=1, bias=False),IWT())

        self.reduce_channel2=ReduceChannel(dim=int(self.num_features*2*2**1))
        
        i_layer = i_layer-1 #128
        self.encoder2=nn.Sequential(
                                BasicLayerDT2(
                                    num_layers=depths[i_layer],
                                    dim=int(self.num_features*2**1),
                                    n_iter=n_iter[i_layer],
                                    stoken_size=to_2tuple(stoken_size[i_layer]),   
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],       
                                    layerscale=layerscale[i_layer],
                                    init_values=init_values))

        self.up1 = nn.Sequential(nn.Conv2d(self.num_features*2**1, self.num_features*2*2**1, kernel_size=3, stride=1, padding=1, bias=False),IWT())

        self.reduce_channel1=ReduceChannel(dim=int(self.num_features*2))
        i_layer = i_layer-1
        self.encoder1=nn.Sequential(
                                BasicLayerDT2(
                                    num_layers=depths[i_layer],
                                    dim=int(self.num_features),
                                    n_iter=n_iter[i_layer],
                                    stoken_size=to_2tuple(stoken_size[i_layer]),   
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],       
                                    layerscale=layerscale[i_layer],
                                    init_values=init_values))

        self.decoder = Decoder(inplanes=int(self.num_features))
        if config.DENOISE:
            self.inversion = Inversion(config)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # print(x.shape)
        
        # print("Deblending Transformer v2 block 1")
        dec1 = self.decorder1(x)
        # print(dec1.shape)
        
        # print("down sampling 1")
        ds1 = self.down1(dec1)
        # print(ds1.shape)
        
        # print("Deblending Transformer v2 block 2")
        dec2 = self.decorder2(ds1)
        # print(dec2.shape)
        
        # print("down sampling 2")
        ds2 = self.down2(dec2)
        # print(ds2.shape)
        
        # print("Deblending Transformer v1 block cen")
        cen = self.center(ds2)
        # print(cen.shape)
        
        # print("up sampling 2")
        us2 = self.up2(cen)
        # print(cen.shape)
        
        # print("up sampling 2 : cat")
        cat2 = torch.cat([dec2, us2], 1)
        # print(cat2.shape)
        
        # print("up sampling 2 : channel/2")
        reduce2 = self.reduce_channel2(cat2)
        # print(reduce2.shape)
        
        # print("Deblending Transformer v2 block 2")
        enc2 = self.encoder2(reduce2)
        # print(enc2.shape)
        
        # print("up sampling 1")
        us1 = self.up1(enc2)
        # print(us1.shape)
        
        # print("up sampling 1 : cat")
        cat1 = torch.cat([dec1, us1], 1)
        # print(cat1.shape)
        
        # print("up sampling 1 : channel/2")
        reduce1 = self.reduce_channel1(cat1)
        # print(reduce1.shape)
        
        # print("Deblending Transformer v2 block 1")
        enc1 = self.encoder1(reduce1)
        # print(enc1.shape)
        return enc1

    def forward(self, x1, index=None):
        # print(x1.shape)
        if self.config.DENOISE:
            count=0
            Flag=True
            while Flag:
                if count == 0:
                    index = torch.cat((index,index),axis=0)
                x1 = self.forward_feature(x1)
                x1 = self.inversion(x1, index)
                loss = x1[-1,0,0,0]
                x1 = x1[:-1,:,:,:]
                if count == 0:
                    min_loss = loss
                else:
                    if loss >= min_loss:
                        Flag = False
                    else:
                        min_loss = loss
                count+=1
            return x1
        else:
            shortcut = x1
            x1 = self.patch_embed(x1)
            x1 = self.forward_features(x1)
            x1 = self.decoder(x1)
            return x1 + shortcut