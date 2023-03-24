import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, weight):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        loss = torch.mean(self.weight * (input - target) ** 2)
        return loss
