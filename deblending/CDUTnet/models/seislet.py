import torch
import torch.nn as nn
from seislet.seislet import  inverse_seislet
from .mobilenetv2 import MobileNetV2

class Seislet(nn.Module):
    def __init__(self,dim,hidden_dim=64,configs=None):
        super(Seislet, self).__init__()
        self.in_feature = MobileNetV2(width_mult=0.25)
    def forward(self,x,indexs): ## index B H W

        B,H,W = indexs.shape
        index = indexs[:,0,:3] # index B 3
        index = index.view(B,3)

        shortcut=x
        x = self.in_feature(x)
        x = x.reshape(B,1,1,1)

        magnitude = torch.absolute(shortcut)
        thresholded = (1 - x / magnitude)
        thresholded = thresholded.clip(min=0, max=None)
        thresholded = shortcut * thresholded
        return thresholded