import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from data_loader import load_npz
from utils import load_data, accuracy, normalize_adj, feature_tensor_normalize, sparse_mx_to_torch_sparse_tensor
from irls_models import IRLS
from tqdm import trange

par = argparse.ArgumentParser()
par.add_argument('--runs', type=int, default=100, help='The number of experiments.')
par.add_argument('--noise_type', type=str, default="gaussian")
par.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')
par.add_argument('--ber', type=float, default=0.1, help='The level of noise')

# ---------------------- Universal ----------------------
par.add_argument("--seed"         , type = int  , default = 42)

# ---------------------- Data ----------------------
par.add_argument("--data"         , type = str  , default = "cora")

# ---------------------- Model ----------------------
par.add_argument("--hidden_size"  , type = int  , default = 128)
par.add_argument("--norm"         , type = str  , default = "none")
par.add_argument("--dropout"      , type = float, default = 0.0)
par.add_argument("--inp_dropout"  , type = float, default = 0.0)

par.add_argument("--mlp_bef"  , type = int  , default = 1)
par.add_argument("--mlp_aft"  , type = int  , default = 0)

# propagation
par.add_argument("--no_precond"   , action = "store_true" , default = False)
par.add_argument("--prop_step"    , type = int  , default = 2)
par.add_argument("--alp"          , type = float, default = 0)  # 0 for alpha = 1 / (1 + lambda)
par.add_argument("--lam"          , type = float, default = 1)

# ---------------------- Train & Test ----------------------
par.add_argument("--num_epoch"    , type = int  , default = 500)
par.add_argument("--patience"     , type = int  , default = -1)
par.add_argument("--lr"           , type = float, default = 1e-3)
par.add_argument("--weight_decay" , type = float, default = 0)

args = par.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load data
if args.data in ['cora', 'citeseer', 'pubmed']: #Cora
    adj, features, idx_train, idx_val, idx_test, labels = load_data(args.data)
else: #coauthor-cs
    features, labels, adj, train_mask, val_mask, test_mask = load_npz(args.data)
    idx_train = np.nonzero(train_mask)[0]
    idx_val = np.nonzero(val_mask)[0]
    idx_test = np.nonzero(test_mask)[0]

# Normalize adj and features
features = features.toarray()
features_array = deepcopy(features)
sem_adj = normalize_adj(adj+sp.eye(adj.shape[0])) 
rowsum = np.array(sp.coo_matrix(adj).sum(1))
d_inv_sqrt = np.power(rowsum, -1.0).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
norm_diag = sp.diags(d_inv_sqrt)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1].to(device)
features = torch.FloatTensor(features).to(device)
sem_adj = sparse_mx_to_torch_sparse_tensor(sem_adj).to(device)
norm_diag = sparse_mx_to_torch_sparse_tensor(norm_diag).to(device)
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
    model = IRLS(input_d=features.shape[1], 
                 output_d=labels.max().item() + 1, 
                 hidden_d=args.hidden_size, 
                 prop_step=args.prop_step, 
                 num_mlp_before=args.mlp_bef, 
                 num_mlp_after=args.mlp_aft, 
                 norm=args.norm, 
                 alp=args.alp, 
                 lam=args.lam, 
                 dropout=args.dropout, 
                 inp_dropout=args.inp_dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    
    min_loss = float('inf')
    bad_counter = 0
    final_val = 0
    final_test = 0
    for epoch in range(args.num_epoch):
        model.train()
        optimizer.zero_grad()
        output = model(noise_features, sem_adj, norm_diag)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(noise_features, sem_adj, norm_diag)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if loss_val < min_loss:
            min_loss = loss_val
            final_val = acc_val
            final_test = acc_test
            bad_counter = 0
        else:
            bad_counter += 1
            if args.patience > 0 and bad_counter >= args.patience:
                break
        

    all_val.append(final_val.item())
    all_test.append(final_test.item())

print(100*np.mean(all_val), 100*np.std(all_val), 100*np.mean(all_test), 100*np.std(all_test))

