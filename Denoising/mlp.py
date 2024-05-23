import argparse
import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from data_loader import load_npz
from utils import load_data, accuracy, feature_tensor_normalize
from gcn.models import MLP
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--noise_type', type=str, default="gaussian")
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')
parser.add_argument('--ber', type=float, default=0.1, help='The level of noise')

parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--early_stopping', type=int, default=10, help='Tolerance for early stopping (# of epochs).')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

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
for run in trange(args.runs, desc='Run Train'):
    noise_features = noise_features_list[run]

    # Model and optimizer
    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    min_loss = float('inf')
    final_val = 0
    final_test = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(noise_features)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        output = model(noise_features)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if loss_val < min_loss:
            min_loss = loss_val
            final_val = acc_val
            final_test = acc_test

    all_val.append(final_val.item())
    all_test.append(final_test.item())

print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

