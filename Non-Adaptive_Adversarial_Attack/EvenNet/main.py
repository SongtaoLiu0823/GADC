import argparse
import numpy as np
from models import EvenNet
from scipy.sparse import load_npz
import torch
from torch_sparse import SparseTensor
from deeprobust.graph.data import Dataset
import torch.nn.functional as F
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--alpha', type=float, default=0.1, help="alpha")
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load dataset
data = Dataset(root='attack_data/', name=args.dataset, seed=42, setting='prognn')
_, features, labels = data.adj, data.features.todense(), data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
adj_mtx = load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))
edge_index = SparseTensor.from_scipy(adj_mtx).float().to(device)
edge_index = edge_index.coo()

rows, cols = edge_index[0], edge_index[1]

edge_index = torch.stack([rows, cols], dim=0)

labels = torch.as_tensor(labels, dtype=torch.long).to(device)
x = torch.from_numpy(features).to(device)
d = x.shape[1]
c = labels.max().item() + 1
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

patience = 30
early_stopping = patience
best_loss_val = 100
all_val_acc = []
all_test_acc = []
for run in trange(args.runs):
    model = EvenNet(in_channels=d, out_channels=c, hidden_size=64, K=10, alpha=args.alpha, Init='PPR', dprate=0.5, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    optimizer = torch.optim.Adam(
                [{'params': model.lin1.parameters(), 'weight_decay': 0.0005, 'lr': 0.01},
                 {'params': model.lin2.parameters(), 'weight_decay': 0.0005, 'lr': 0.01},
                 {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': 0.01}])
    
    best_val = float('-inf')
    final_test_acc = 0
    final_val_acc = 0
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(x, edge_index)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        preds = output[idx_val].max(1)[1].type_as(labels)
        correct = preds.eq(labels[idx_val]).double()
        correct = correct.sum()
        val_acc = correct / len(labels[idx_val])

        if best_loss_val > loss_val:
            best_loss_val = loss_val
            preds = output[idx_test].max(1)[1].type_as(labels)
            correct = preds.eq(labels[idx_test]).double()
            correct = correct.sum()
            final_val_acc = val_acc
            final_test_acc = correct / len(labels[idx_test])
            
        else:
            patience -= 1
        if epoch > early_stopping and patience <= 0:
            break
    if final_test_acc > 0:
        all_val_acc.append(final_val_acc.item())
        all_test_acc.append(final_test_acc.item())
        print(len(all_test_acc))
    if len(all_test_acc) == 10:
        break
print(np.mean(all_val_acc), np.std(all_val_acc), np.mean(all_test_acc), np.std(all_test_acc))

