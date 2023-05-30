import os
import torch
from torch import nn
from torch.autograd import Function


def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        dx = -grads.new_tensor(1) * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return GradientReversalFunction.apply(x)