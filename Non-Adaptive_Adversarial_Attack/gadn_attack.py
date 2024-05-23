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
from gcn.models import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')


parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--lam', type=float, default=1.0, help='lam.')
parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon.')
parser.add_argument('--degree', type=int, default=4, help='degree of the approximation.')

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

def get_adversarial_transition(features, adj, epsilon):
    features_center = features[row]
    features_neighbor = features[col]
    features_center_norm = np.linalg.norm(features_center, axis=1)
    features_neighbor_norm = np.linalg.norm(features_neighbor, axis=1)
    norm_dot = features_center_norm * features_neighbor_norm
    cos_values = np.sum(features_center * features_neighbor, axis=1) / norm_dot


    #features_center = features[row]
    #features_neighbor = features[col]
    #attack_values = np.sum(features_center * features_neighbor, axis=1)
    #attack_values = epsilon * attack_values / np.linalg.norm(attack_values)
    attack_adj = sp.coo_matrix((cos_values, (row, col)), shape=(features.shape[0], features.shape[0]))
    #attack_adj = sp.coo_matrix((attack_values, (row, col)), shape=(features.shape[0], features.shape[0]))
    attack_adj = sparse_mx_to_torch_sparse_tensor(attack_adj).cuda()
    #attack_adj = sparse_mx_to_torch_sparse_tensor(attack_adj + sp.eye(adj.shape[0])).cuda()
    #final_adj = adj - attack_adj
    return attack_adj

def graph_diffusion(output, adj, degree, lam):
    emb = output
    neumann_adj = adj * lam / (1+lam)
    for _ in range(degree):
        output = torch.spmm(neumann_adj, output)
        emb += output
    return 1/(1+lam) * emb


args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = Dataset(root='attack_data/', name=args.dataset, seed=args.seed, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj = sp.load_npz("meta/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))

indices = np.nonzero(adj)
row, col = indices[1], indices[0]


labels = torch.LongTensor(labels).to(device)
n_features = torch.FloatTensor(normalize_features(features.toarray())).to(device)
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

adversarial_transition = get_adversarial_transition(features.toarray(), adj_normalized, args.epsilon)
diff_features = graph_diffusion(n_features, adversarial_transition, args.degree, args.lam)
all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    #features = Neumann_precompute(n_features, attack_adj, args.degree, args.lam, args.epsilon)

    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout).to(device)
    #model = SGC(features.shape[1], labels.max().item() + 1).to(device)
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
            best_acc_val = loss_val
            final_val = acc_val.item()
            final_test = acc_test.item()

    all_val_acc.append(final_val)
    all_test_acc.append(final_test)

print(100*np.mean(all_val_acc), np.std(all_val_acc), 100*np.mean(all_test_acc), 100*np.std(all_test_acc))


