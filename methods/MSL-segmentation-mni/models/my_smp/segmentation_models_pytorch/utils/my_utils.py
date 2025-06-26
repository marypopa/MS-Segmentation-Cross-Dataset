import torch
import torch.nn as nn
from .base import Loss
import numpy as np
from skimage import measure, morphology

class WeightedBCELoss(Loss):
    def __init__(self, weight=None):
        super(WeightedBCELoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        # Apply sigmoid activation to the model output

        # Compute the binary cross-entropy loss
        eps_clip = 1e-7
        output = torch.clamp(output, min=eps_clip, max=1 - eps_clip)
        loss = - (self.weight * target * torch.log(output) + (1 - self.weight) * (1 - target) * torch.log(1 - output))

        # Take the mean over the batch
        return torch.mean(loss)
