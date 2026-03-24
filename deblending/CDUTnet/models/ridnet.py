import torch.nn as nn
from models import ops


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = ops.BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        #self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        #g = self.g(r3)
        out = self.ca(r3)

        return out
        


class RIDNET(nn.Module):
    def __init__(self, n_feats):
        super(RIDNET, self).__init__()
        
        self.n_feats = n_feats
        kernel_size = 3

        self.head = ops.BasicBlock(1, n_feats, kernel_size, 1, 1)

        self.b1 = Block(self.n_feats, self.n_feats)
        self.b2 = Block(self.n_feats, self.n_feats)
        self.b3 = Block(self.n_feats, self.n_feats)
        self.b4 = Block(self.n_feats, self.n_feats)

        self.tail = nn.Conv2d(self.n_feats, 1, kernel_size, 1, 1, 1)



    def forward(self, x):

        h = self.head(x)

        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)

        res = self.tail(b_out)
        f_out = res + x 

        return f_out 
