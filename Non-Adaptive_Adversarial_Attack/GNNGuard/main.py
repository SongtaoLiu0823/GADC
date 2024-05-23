import torch
import sys
sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='attack_data/', name=args.dataset, seed=42, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

modified_adj = sp.load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))

attention = args.GNNGuard # if True, our method; if False, run baselines

def test(adj):
    ''' testing model '''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    classifier.eval()

    # classifier.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    acc_test = classifier.test(idx_test)
    return acc_test

def main():


    print('=== testing GCN on Mettacked graph ===')
    all_acc_test = []
    for _ in range(10):
        acc_test = test(modified_adj)
        all_acc_test.append(acc_test)
    print(np.mean(all_acc_test), np.std(all_acc_test))


if __name__ == '__main__':
    main()


