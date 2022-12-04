import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1):

        inputs = torch.sigmoid(inputs)
        inputs[inputs < 0.5] = 0
        inputs[inputs > 0.5] = 1       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = torch.sum(inputs * targets)
        A = torch.sum(inputs )
        B = torch.sum(targets)
        dice = (2 * intersection + eps) / (A + B + eps)
        
        return 1 - dice