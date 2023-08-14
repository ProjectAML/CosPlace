import torch
import torch.nn as nn
from torch.nn import Parameter

class CenterLoss(nn.Module):
    def __init__(self, num_class, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_class, feat_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x, y):
        # Calculate the centers associated with the class labels
        centers_batch = self.centers.index_select(0, y)

        # Calculates the Euclidean distance between features and centers
        loss = torch.sum(torch.pow(x - centers_batch, 2), dim=1).mean()

        # Update centers based on current examples
        diff = centers_batch - x
        unique_labels = y.unique()
        counts = y.bincount(minlength=self.num_class).float().unsqueeze(1)
        self.centers[unique_labels] -= self.alpha * (diff / counts[unique_labels])
        return loss
