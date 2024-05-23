from collections import OrderedDict
import torch

import sys
sys.path.append("..")
import gb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer'], help='dataset')
args = parser.parse_args()

dataset = args.dataset
A, X, y = gb.data.get_dataset(dataset)
N, D = X.shape
C = y.max().item() + 1
train_nodes, val_nodes, test_nodes = gb.data.get_splits(y)[0]  # [0] = select first split

A = A.cuda()
X = X.cuda()
y = y.cuda()
print(A, X, y, train_nodes, val_nodes, test_nodes)
torch.manual_seed(42)

rank = 50
fit_kwargs = dict(lr=1e-2, weight_decay=5e-4)
budget = 1000

def make_model(div_limit=1e-6):
    return gb.model.GNNGuard(n_feat=D, n_class=C, hidden_dims=[16], dropout=0.5, div_limit=div_limit).cuda()

aux_model = make_model()
aux_model.fit((A, X), y, train_nodes, val_nodes, progress=False, **fit_kwargs)

def loss_fn(A_flip):
    A_diff = A_flip * (1 - 2 * A)
    A_pert = A + A_diff

    ########## w/ real div_limit #########
    # alteration = dict()
    ######## w/ changed div_limit ########
    alteration = dict(div_limit=1e-2)
    ######################################

    ############# Aux-Attack #############
    with gb.model.changed_fields(aux_model, **alteration):
        scores = aux_model(A_pert, X)
    ############# Meta-Attack ############
    # model = make_model(**alteration)
    # model.fit((A_pert, X), y, train_nodes, val_nodes, progress=False, **fit_kwargs, max_epochs=50, differentiable=A_pert.requires_grad)
    # scores = model(A_pert, X)
    ######################################

    return gb.metric.margin(scores[test_nodes, :], y[test_nodes]).tanh().mean()

def grad_fn(A_flip):
    return torch.autograd.grad(loss_fn(A_flip), A_flip)[0]

########### FGA for Aux-Attack ###########
# pert = gb.attack.greedy_grad_descent(A.shape, True, A.device, [budget], grad_fn, flips_per_iteration=budget, max_iterations=1)[0]
########### PGD for Aux-Attack ###########
pert, _ = gb.attack.proj_grad_descent(A.shape, True, A.device, budget, grad_fn, loss_fn, base_lr=0.1)
######### Greedy for Meta-Attack #########
# pert = gb.attack.greedy_grad_descent(A.shape, True, A.device, [budget], grad_fn, flips_per_iteration=1)[0]
########### PGD for Meta-Attack ##########
# pert, _ = gb.attack.proj_grad_descent(A.shape, True, A.device, budget, grad_fn, loss_fn, base_lr=0.01, grad_clip=1)
##########################################

print("Clean test acc:   ", gb.metric.accuracy(aux_model(A, X)[test_nodes], y[test_nodes]).item())

A_pert = A + gb.pert.edge_diff_matrix(pert, A)
print("Adversarial edges:", pert.shape[0])
print("Evasion test acc: ", gb.metric.accuracy(aux_model(A_pert, X)[test_nodes], y[test_nodes]).item())

pois_model = make_model()
pois_model.fit((A_pert, X), y, train_nodes, val_nodes, progress=False, **fit_kwargs)
print("Poisoned test acc:", gb.metric.accuracy(pois_model(A_pert, X)[test_nodes], y[test_nodes]).item())
