import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.layers import GraphConvolution, MLPLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        return x

