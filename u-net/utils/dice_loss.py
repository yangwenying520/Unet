import torch
import torch.nn as nn
import numpy as np
from skimage import measure,color
import cv2
"""Similar to IOU """
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        # print(target.shape)
        intersection = input_flat * target_flat

        loss = (2 * (intersection.sum(1)) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # print(loss.shape)
        # print(loss.shape)
        loss = 1-loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class loss_weight():
    def __init__(self, w0=10, sigma=5, n_classes=2):
        self.w0 = w0
        self.sigma = sigma
        self.n_classes = n_classes

    def class_weight(self,mask=None):
        c_weights = np.zeros((mask.shape[0],self.n_classes))
        for i in range(mask.shape[0]):
            for j in range(self.n_classes):
                c_weights[i,j] = 1.0 / ((mask[i,:,:] == j).sum())
            c_weights[i,:] /= c_weights[i,:].max()
        cw_map = np.zeros((mask.shape[0],self.n_classes,mask.shape[1],mask.shape[2]))
        for i in range(mask.shape[0]):
            for j in range(self.n_classes):
                cw_map[i,j,:,:] = c_weights[i,j]
        # print(cw_map[0])
        return cw_map








