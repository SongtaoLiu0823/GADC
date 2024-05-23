import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from utils import accuracy, normalize_features, sparse_mx_to_torch_sparse_tensor
from tqdm import trange
from deeprobust.graph.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')
parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--lam', type=float, default=1.0, help='lam.')
parser.add_argument('--degree', type=int, default=6, help='degree of the approximation.')


class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


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

def graph_diffusion(output, adj, degree, lam):
    emb = output
    neumann_adj = adj * lam / (1+lam)
    for _ in range(degree):
        output = torch.spmm(neumann_adj, output)
        emb += output
    return 1/(1+lam) * emb


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load data
data = Dataset(root='attack_data/', name=args.dataset, seed=args.seed, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj = sp.load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))

'''Recover the adjacency matrix'''
features = features.toarray()
indices = np.nonzero(adj)
row, col = indices[0], indices[1]
features_center = features[row]
features_neighbor = features[col]
features_center_norm = np.linalg.norm(features_center, axis=1)
features_neighbor_norm = np.linalg.norm(features_neighbor, axis=1)
norm_dot = features_center_norm * features_neighbor_norm
cos_values = np.sum(features_center * features_neighbor, axis=1) / norm_dot
recover_adj = sp.coo_matrix((cos_values, (row, col)), shape=(features.shape[0], features.shape[0]))
recover_adj = recover_adj + sp.eye(adj.shape[0])
recover_adj = sparse_mx_to_torch_sparse_tensor(recover_adj).cuda()


labels = torch.LongTensor(labels).to(device)
n_features = torch.FloatTensor(normalize_features(features)).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

diff_features = graph_diffusion(n_features, recover_adj, args.degree, args.lam)
all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Train'):
    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    final_val = 0
    final_test = 0
    best_loss_val = np.inf
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output = torch.log_softmax(model(diff_features), dim=-1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(diff_features)
        output = torch.log_softmax(output, dim=1)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
            
        if best_loss_val > loss_val:
            best_loss_val = loss_val
            final_val = acc_val.item()
            final_test = acc_test.item()

    all_val_acc.append(final_val)
    all_test_acc.append(final_test)

print(np.mean(all_val_acc), np.std(all_val_acc), np.mean(all_test_acc), np.std(all_test_acc))

