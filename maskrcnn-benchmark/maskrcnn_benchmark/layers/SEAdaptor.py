import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAdaptor(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SEAdaptor, self).__init__()
        if n_channels % reduction != 0:
            raise ValueError('n_channels must be divisible by reduction (default = 16)')
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                    nn.Linear(n_channels, n_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_channels // reduction, n_channels),
                    nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y