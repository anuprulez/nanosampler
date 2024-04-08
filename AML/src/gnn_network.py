import torch
from torch_geometric.data import Data
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import pandas as pd
import numpy as np


class GCN(torch.nn.Module):
    '''
    Neural network with graph convolution network (GCN)
    '''
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.batch_norm2 = BatchNorm1d(2 * hidden_dim)
        self.batch_norm3 = BatchNorm1d(hidden_dim)
        self.batch_norm4 = BatchNorm1d(hidden_dim // 2)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.batch_norm1(F.relu(h))
        h = self.conv2(h, edge_index)
        h = self.batch_norm2(F.relu(h))
        h = self.conv3(h, edge_index)
        h = self.batch_norm3(F.relu(h))
        h = self.conv4(h, edge_index)
        h = self.batch_norm4(F.relu(h))
        out = self.classifier(h)
        return out