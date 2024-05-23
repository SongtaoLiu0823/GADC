import argparse
import numpy as np
import random
import torch.nn as nn
import torch
import torch_sparse
import torch.nn.functional as F
import torch_geometric.transforms as T
import time
from tqdm import trange
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn.dense.linear import Linear
from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--lam', type=float, default=32, help='lam.')
parser.add_argument('--degree', type=int, default=128, help='degree of the approximation.')
parser.add_argument('--epsilon', type=float, default=0.01, help='epsilon.')

args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.bias_list = []
        self.lins.append(Linear(in_channels, hidden_channels, bias=False,
                          weight_initializer='glorot'))
        self.bias_list.append(Parameter(torch.Tensor(hidden_channels)).to(device))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels, bias=False,
                          weight_initializer='glorot'))
            self.bias_list.append(Parameter(torch.Tensor(hidden_channels)).to(device))
        self.lins.append(Linear(hidden_channels, out_channels, bias=False,
                          weight_initializer='glorot'))
        self.bias_list.append(Parameter(torch.Tensor(out_channels)).to(device))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bias in self.bias_list:
            zeros(bias)

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = x + self.bias_list[i]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x) + self.bias_list[-1]
        return torch.log_softmax(x, dim=-1)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


dataset = PygNodePropPredDataset(name='ogbn-products', transform=T.ToSparseTensor())
data = dataset[0]
split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)
# Pre-compute GCN normalization.
adj_t = data.adj_t.set_diag()
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
data.adj_t = adj_t

data = data.to(device)

row, col, edge_attr = data.adj_t.t().coo()

def Neumann_precompute(features, adj, degree, lam, epsilon):
    emb = features
    num_nodes = adj.size(0)
    size = int(row.shape[0]/10)
    values = []
    for i in trange(0, 11):
        row_sample = row[i*size:min((i+1)*size, row.shape[0])]
        col_sample = col[i*size:min((i+1)*size, row.shape[0])]
        features_center = features[row_sample]
        features_neighbor = features[col_sample]
        value_sample = torch.sum(torch.mul(features_center, features_neighbor), dim=1)
        values.append(value_sample)
    value = torch.cat(values)
    value_norm = torch.norm(value)
    added_adj = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_nodes, num_nodes))
    t = 1
    for i in range(degree):
        t = t * lam / (1+lam)
        features = torch_sparse.matmul(adj, features) - args.epsilon / value_norm * torch_sparse.matmul(added_adj, features)
        emb = emb + t*features
    return 1/(1+lam) * emb

model = MLP(data.num_features, args.hidden_channels,
            dataset.num_classes, args.num_layers,
            args.dropout).to(device)

# noise
noise_features_list = []
for _ in trange(args.runs):
    noise_features = data.x + args.noise_level * torch.randn(size=data.x.shape, device=device)
    noise_features_list.append((noise_features))


evaluator = Evaluator(name='ogbn-products')
logger = Logger(args.runs, args)

for run in range(args.runs):
    noise_features = noise_features_list[run]
    noise_features = Neumann_precompute(noise_features, data.adj_t, args.degree, args.lam, args.epsilon)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()

        output = model(noise_features)
        loss_train = F.nll_loss(output[train_idx], data.y.squeeze(1)[train_idx])
        
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(noise_features)
            y_pred = output.argmax(dim=-1, keepdim=True)

            train_acc = evaluator.eval({
                'y_true': data.y[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
            valid_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']

            logger.add_result(run, (train_acc, valid_acc, test_acc))
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss_train.item():.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
    logger.print_statistics(run)
logger.print_statistics()



