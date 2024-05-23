import argparse
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange
from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true', default=False, help='test the performance of gcn without other components')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False, help='whether use symmetric matrix')
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
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout, device=device)
    model = model.to(device)

    adj_, features_, labels_ = preprocess(adj, features, labels, preprocess_adj=False, device=device)

    prognn = ProGNN(model, args, device)
    prognn.fit(features_, adj_, labels_, idx_train, idx_val)
    acc_test = prognn.test(features_, labels_, idx_test)
    all_test_acc.append(acc_test)

print(np.mean(all_test_acc), np.std(all_test_acc))

