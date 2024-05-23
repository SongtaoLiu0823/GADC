import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from gcn.models import MLP
from tqdm import trange
from utils import load_new_data, accuracy, normalize_adj, feature_tensor_normalize, sparse_mx_to_torch_sparse_tensor, split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lam', type=float, default=1.0, help='lam.')
parser.add_argument('--weight_decay', type=float, default=5e-04, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cornell", help='Dataset to use.')
parser.add_argument('--degree', type=int, default=16, help='degree of the approximation.')
parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon.')

args = parser.parse_args()

def get_adversarial_transition(features, adj, epsilon):
    features_center = features[row]
    features_neighbor = features[col]
    features_center_norm = np.linalg.norm(features_center, axis=1)
    features_neighbor_norm = np.linalg.norm(features_neighbor, axis=1)
    norm_dot = features_center_norm * features_neighbor_norm
    attack_values = epsilon * np.sum(features_center * features_neighbor, axis=1) / norm_dot
    attack_adj = sp.coo_matrix((attack_values, (row, col)), shape=(features.shape[0], features.shape[0]))
    attack_adj = sparse_mx_to_torch_sparse_tensor(attack_adj).cuda()
    final_adj = adj - attack_adj
    return final_adj

def graph_diffusion(output, adj, degree, lam):
    emb = output
    neumann_adj = adj * lam / (1+lam)
    for _ in range(degree):
        output = torch.spmm(neumann_adj, output)
        emb += output
    return 1/(1+lam) * emb


# setting random seeds
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj, features, labels = load_new_data(args.dataset)

indices = np.nonzero(adj)
row, col = indices[1], indices[0]

# Normalize adj and features
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1].to(device)
features = torch.FloatTensor(features).to(device)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)

# noise
features_list = []
for _ in trange(args.runs):
    features_list.append(feature_tensor_normalize(features))

all_val = []
all_test = []
for run in trange(args.runs, desc='Run Train'):
    idx_train, idx_val, idx_test = split_dataset(labels.size()[0])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    cur_features = features_list[run]

    adversarial_transition = get_adversarial_transition(cur_features.cpu().numpy(), adj_normalized, args.epsilon)
    diff_features = graph_diffusion(cur_features, adversarial_transition, args.degree, args.lam)

    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc_val = 0
    final_val = 0
    final_test = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(diff_features)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output = model(diff_features)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                final_val = acc_val
                final_test = acc_test    
    all_val.append(final_val.item())
    all_test.append(final_test.item())

print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

