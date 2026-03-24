import torch
from torch import nn as nn
from torch.nn import functional as F

class L1(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1, self).__init__()
        self.eps = 1e-6

    def forward(self, X1, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss 

class MIX2(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(MIX2, self).__init__()
        self.eps = 1e-6

    def forward(self, X1, X2, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return 0.5*loss + 0.5*F.mse_loss(X2,Y)

class MIX3(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(MIX3, self).__init__()
        self.eps = 1e-6

    def forward(self, X1,X2, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss + F.mse_loss(X2,Y)

class Mixloss(nn.Module):
    def __init__(self, weight1=0.5,weight2=0.5, reduction='mean'):
        super(Mixloss, self).__init__()
        self.reduction = reduction
        self.weight1=weight1
        self.weight2=weight2


    def forward(self,pred,target):
        pred01 = pred[:, :, 0::2, :] / 2
        pred02 = pred[:, :, 1::2, :] / 2
        pred1 = pred01[:, :, :, 0::2]
        pred2 = pred02[:, :, :, 0::2]
        pred3 = pred01[:, :, :, 1::2]
        pred4 = pred02[:, :, :, 1::2]
        pred_HL = -pred1 - pred2 + pred3 + pred4 ##vertical
        pred_LH = -pred1 + pred2 - pred3 + pred4  ## hozational

        target01 = target[:, :, 0::2, :] / 2
        target02 = target[:, :, 1::2, :] / 2
        target1 = target01[:, :, :, 0::2]
        target2 = target02[:, :, :, 0::2]
        target3 = target01[:, :, :, 1::2]
        target4 = target02[:, :, :, 1::2]
        target_HL = -target1 - target2 + target3 + target4 ##vertical
        target_LH = -target1 + target2 - pred3 + target4  ## hozational

        return F.mse_loss(pred,target)+self.weight1*F.l1_loss(pred_HL, target_HL)+self.weight2*F.l1_loss(pred_LH, target_LH)
