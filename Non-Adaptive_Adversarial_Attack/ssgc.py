import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from tqdm import trange
from deeprobust.graph.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')
parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.05, help='alpha.')
parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--lam', type=float, default=1.0, help='lam.')
parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)
        # self.bn = nn.BatchNorm1d(nfeat)

    def forward(self, x):
        return self.W(x)

def sgc_precompute(features, adj, degree, alpha):
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    return emb


args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Load data
data = Dataset(root='attack_data/', name=args.dataset, seed=15, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

features = features.toarray()
attack_adj = sp.load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))
n_features = normalize_features(features)

labels = torch.LongTensor(labels).to(device)
n_features = torch.FloatTensor(n_features).to(device)
attack_adj = normalize_adj(attack_adj + sp.eye(attack_adj.shape[0]))
attack_adj = sparse_mx_to_torch_sparse_tensor(attack_adj).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)


all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    features = sgc_precompute(n_features, attack_adj, args.degree, args.alpha)
    model = SGC(features.shape[1], labels.max().item() + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model

    best_epoch = 0
    final_val = 0
    final_test = 0
    min_loss_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output = torch.log_softmax(model(features), dim=-1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(features)
        output = torch.log_softmax(output, dim=1)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if loss_val < min_loss_val:
            min_loss_val = loss_val
            final_val = acc_val.item()
            final_test = acc_test.item()

    
    all_val_acc.append(final_val)
    all_test_acc.append(final_test)
    
print(np.mean(all_val_acc), np.std(all_val_acc), np.mean(all_test_acc), np.std(all_test_acc))

