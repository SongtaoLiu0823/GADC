import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from data_loader import load_npz
from copy import deepcopy
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from tqdm import trange
from utils import load_data, accuracy, feature_tensor_normalize


# Training settings
exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--noise_type', type=str, default="gaussian")
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')
parser.add_argument('--ber', type=float, default=0.1, help='The level of noise')

parser.add_argument('--heads_2', type=int, default=1)
parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=args.heads_2, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index), alpha=0.2)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Load data
if args.dataset in ['cora', 'citeseer', 'pubmed']: #Cora
    adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
    data = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())[0]
    data = data.to(device)
    edge_index = data.edge_index
else: #coauthor-cs
    features, labels, adj, train_mask, val_mask, test_mask = load_npz(args.dataset)
    edge_index = torch.LongTensor(np.nonzero(adj)).to(device)
    idx_train = np.nonzero(train_mask)[0]
    idx_val = np.nonzero(val_mask)[0]
    idx_test = np.nonzero(test_mask)[0]


# Normalize adj and features
features = features.toarray()
features_array = deepcopy(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1].to(device)
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
for run in trange(args.runs, desc='Run Experiments'):
    noise_features = noise_features_list[run]

    # Model and optimizer
    model = GAT(in_channels=features.shape[1], 
                out_channels=int(labels.max().item()) + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    acc_val_max = 0
    loss_val_min = np.inf
    final_model = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = torch.log_softmax(model(noise_features, edge_index), dim=-1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = torch.log_softmax(model(noise_features, edge_index), dim=-1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()

        if acc_val >= acc_val_max or loss_val <= loss_val_min:
            if acc_val >= acc_val_max and loss_val <= loss_val_min:
                final_model = deepcopy(model) 
            acc_val_max = np.max((acc_val, acc_val_max))
            loss_val_min = np.min((loss_val, loss_val_min))
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == args.patience:
                break
    final_model.eval()
    output = torch.log_softmax(final_model(noise_features, edge_index), dim=-1)
    final_val = accuracy(output[idx_val], labels[idx_val]).item()
    final_test = accuracy(output[idx_test], labels[idx_test]).item()
    all_val.append(final_val)
    all_test.append(final_test)

print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

