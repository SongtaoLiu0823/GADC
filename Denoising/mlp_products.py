import argparse
import numpy as np
import random
from tqdm import trange
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)



parser = argparse.ArgumentParser(description='OGBN-Products (MLP)')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--noise_level', type=float, default=0.1, help='The level of noise')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--runs', type=int, default=10)


args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-products')
data = dataset[0]

data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)

# noise
noise_features_list = []
for _ in trange(args.runs):
    noise_features = data.x + args.noise_level * torch.randn(size=data.x.shape, device=device)
    noise_features_list.append((noise_features))

model = MLP(data.num_features, args.hidden_channels,
            dataset.num_classes, args.num_layers,
            args.dropout).to(device)

evaluator = Evaluator(name='ogbn-products')
logger = Logger(args.runs, args)

for run in range(args.runs):
    noise_features = noise_features_list[run]
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


