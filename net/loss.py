"""
This code is inspired from:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/nnUNet/nnunet/loss.py
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
import numpy as np


def compute_loss(preds, label, criterion, deep_supervision=False):
    if deep_supervision:
        loss, weights = 0.0, 0.0
        for i, pred in enumerate(preds):
            loss += criterion(pred, label) * 0.5 ** i
            weights += 0.5 ** i
        return loss / weights
    
    return criterion(preds, label)

class LossBraTS(nn.Module):
    def __init__(self, focal):
        super(LossBraTS, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False) if focal else nn.BCEWithLogitsLoss()

    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y.float())

    def forward(self, p, y):
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        p_wt, p_tc, p_et = p[:, 0].unsqueeze(1), p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1)
        l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et)
        return l_wt + l_tc + l_et

class DiceCoeff(nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
    
    def forward(self, pred, y):
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        p_wt, p_tc, p_et = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1), pred[:, 2].unsqueeze(1)   
        l_wt, l_tc, l_et = self.dice(p_wt, y_wt), self.dice(p_tc, y_tc), self.dice(p_et, y_et)
<<<<<<< HEAD
        return (1-l_wt), (1-l_tc), (1-l_et)
=======

        return np.array([(1-l_wt).cpu(), (1-l_tc).cpu(), (1-l_et).cpu()])
>>>>>>> 274ab5c3c134b6b3a365910f2b1241d575cb05db
