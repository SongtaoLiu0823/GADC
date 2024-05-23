import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import trange

from copy import deepcopy
from data_loader import load_npz
from utils import accuracy, load_data, normalize_adj, sparse_mx_to_torch_sparse_tensor, feature_tensor_normalize

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--noise_type', type=str, default="gaussian")
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')
parser.add_argument('--ber', type=float, default=0.1, help='The level of noise')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.05, help='alpha.')
parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora", help='Dataset to use.')
parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')

args = parser.parse_args()


class SGC(torch.nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = torch.nn.Linear(nfeat, nclass)
        # self.bn = torch.nn.BatchNorm1d(nfeat)

    def forward(self, x):
        return self.W(x)

def sgc_precompute(features, adj, degree, alpha):
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    return emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# setting random seeds
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
args.cuda = torch.cuda.is_available()

# Load data
if args.dataset in ['cora', 'citeseer', 'pubmed']: #Cora
    adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
else: #coauthor-cs
    features, labels, adj, train_mask, val_mask, test_mask = load_npz(args.dataset)
    idx_train = np.nonzero(train_mask)[0]
    idx_val = np.nonzero(val_mask)[0]
    idx_test = np.nonzero(test_mask)[0]

# Normalize adj and features
features = features.toarray()
features_array = deepcopy(features)
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1].to(device)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)
features = torch.FloatTensor(features).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

# noise
noise_features_list = []
for _ in trange(args.runs):
    if args.noise_type == "gaussian":
        noise_features = features + args.noise_level * torch.randn(size=features.shape, device=device)
        noise_features_list.append(feature_tensor_normalize(noise_features))
    else:
        flip_features = np.random.binomial(1, args.ber, size=features_array.shape)
        noise_features = (1*np.logical_not(features_array))*flip_features+features_array*(1*np.logical_not(flip_features))
        noise_features_list.append(torch.tensor(noise_features, dtype=torch.float32, device=device))

all_val = []
all_test = []
for run in trange(args.runs, desc='Run Train'):
    features = sgc_precompute(noise_features_list[run], adj_normalized, args.degree, args.alpha)
    model = SGC(features.size(1), labels.max().item()+1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss_val = 100.
    final_val = 0
    final_test = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output = model(features)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                final_val = acc_val
                final_test = acc_test
    if isinstance(final_val, int):
        all_val.append(final_val)
    else:
        all_val.append(final_val.item())

    if isinstance(final_test, int):
        all_test.append(final_test)
    else:
        all_test.append(final_test.item())
print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

