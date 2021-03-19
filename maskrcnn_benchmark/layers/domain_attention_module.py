import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

from .se_module import SELayer

class DomainAttention(nn.Module):
    def __init__(self, in_channels, config, reduction=16):
        super(DomainAttention, self).__init__()
        self.in_channels = in_channels

        num_adapters = config['num_adapters'] # set to 2
        self.n_datasets = num_adapters

        self.fixed_block = config['fixed_block']
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not self.fixed_block and config['less_blocks']:
            if config['block_id'] != 4:
                if config['layer_index'] % 2 == 0:
                    self.fixed_block = True
            else:
                if config['layer_index'] % 2 != 0:
                    self.fixed_block = True
                    
        if self.fixed_block or num_adapters == 1:
            self.SE_Layers = nn.ModuleList([SELayer(in_channels, reduction, with_sigmoid=False) for num_class in range(1)])
        else:
            self.SE_Layers = nn.ModuleList([SELayer(in_channels, reduction, with_sigmoid=False) for num_class in range(num_adapters)])

        self.fc_1 = nn.Linear(in_channels, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()
        
        if self.fixed_block:
            SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            # domain attention: compute the weights for each adapter
            weight = self.fc_1(self.avg_pool(x).view(b, c))
            weight = self.softmax(weight).view(b, self.n_datasets, 1)

            # Create the Universal SE Adapter bank
            for i, SE_Layer in enumerate(self.SE_Layers):
                if i == 0:
                    SELayers_Matrix = SE_Layer(x).view(b, c, 1)
                else:
                    SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)

            # Perform domain assignment based on weights obtained from domain attention step above
            SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)
            # channel importance resclaed to lie in [0, 1]
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)

        # return channel wise multiplication
        return x*SELayers_Matrix