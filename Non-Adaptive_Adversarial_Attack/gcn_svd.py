import torch
import numpy as np
import scipy.sparse as sp
from tqdm import trange
from deeprobust.graph.defense import GCNSVD
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')
parser.add_argument('--k', type=int, default=15, help='Truncated Components.')
parser.add_argument('--runs', type=int, default=10)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data = Dataset(root='attack_data/', name=args.dataset, seed=args.seed, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
adj = sp.load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))


all_test_acc = []
for _ in trange(args.runs):
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
    model = model.to(device)

    model.fit(features, adj, labels, idx_train, idx_val, k=args.k, verbose=False)
    model.eval()

    acc_test = model.test(idx_test)
    all_test_acc.append(acc_test)
print(np.mean(all_test_acc), np.std(all_test_acc))
