import os
import sys
import random
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from copy import deepcopy
from torch_geometric.datasets import Planetoid
from airgnn_model import AirGNN
from tqdm import trange
from utils import load_data, accuracy, feature_tensor_normalize

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# Training settings
exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')

parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lambda_amp', type=float, default=0.1)
parser.add_argument('--lcc', type=str2bool, default=False)
parser.add_argument('--normalize_features', type=str2bool, default=True)
parser.add_argument('--random_splits', type=str2bool, default=False)
#parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
parser.add_argument('--model_cache', type=str2bool, default=False)

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
#data = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())[0]
#data = Planetoid(path, args.dataset, transform=T.ToSparseTensor())[0]
dataset = Planetoid(path, args.dataset)
dataset.transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])

# Normalize adj and features
features = features.toarray()

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1].to(device)
features = torch.FloatTensor(features).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)
#data = data.to(device)
data = dataset[0].to(device)

# noise
noise_features_list = []
for _ in trange(args.runs):
    noise_features = features + args.noise_level * torch.randn(size=features.shape, device=device)
    noise_features_list.append(feature_tensor_normalize(noise_features))

all_val = []
all_test = []
for run in trange(args.runs, desc='Run Experiments'):
    noise_features = noise_features_list[run]

    # Model and optimizer
    model = AirGNN(num_features=features.shape[1], 
                   num_classes=int(labels.max().item()) + 1,
                   args=args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    acc_val_max = 0
    loss_val_min = np.inf
    final_model = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = torch.log_softmax(model(noise_features, data.adj_t), dim=-1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = torch.log_softmax(model(noise_features, data.adj_t), dim=-1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()

        if loss_val <= loss_val_min:
            if loss_val <= loss_val_min:
                final_model = deepcopy(model) 
            loss_val_min = np.min((loss_val, loss_val_min))
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == args.patience:
                break
    final_model.eval()
    output = torch.log_softmax(final_model(noise_features, data.adj_t), dim=-1)
    final_val = accuracy(output[idx_val], labels[idx_val]).item()
    final_test = accuracy(output[idx_test], labels[idx_test]).item()
    all_val.append(final_val)
    all_test.append(final_test)

print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

