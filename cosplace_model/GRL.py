import os
import torch
from torch import nn
from torch.autograd import Function


def discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    #def forward(ctx, x, lambda_):
    #    ctx.lambda_ = lambda_
    #    return x.clone()
    
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    #def backward(ctx, grads):
    #    lambda_ = ctx.lambda_
    #    lambda_ = grads.new_tensor(lambda_)
    #    dx = -lambda_ * grads
    #    return dx, None
    
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