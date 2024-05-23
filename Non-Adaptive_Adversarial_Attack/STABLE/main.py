import torch.nn as nn
import time
import argparse
from utils import *
from model import GCN, LogReg
from copy import deepcopy
import scipy
from robcon import get_contrastive_emb
from deeprobust.graph.data import Dataset, PrePtbDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--threshold', type=float, default=1,  help='threshold')
parser.add_argument('--jt', type=float, default=0.03,  help='jaccard threshold')
parser.add_argument('--cos', type=float, default=0.1,  help='cosine similarity threshold')
parser.add_argument('--k', type=int, default=3 ,  help='add k neighbors')
parser.add_argument('--alpha', type=float, default=0.3,  help='add k neighbors')
parser.add_argument('--beta', type=float, default=2,  help='the weight of selfloop')
parser.add_argument("--log", action='store_true', help='run prepare_data or not')
parser.add_argument('--attack', type=str, default='mettack',  help='attack method')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.log:
    logger = get_logger('./log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
else:
    logger = get_logger('./log/try.log')

if args.attack == 'nettack':
    args.ptb_rate = int(args.ptb_rate)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Loading data
dataset = args.dataset
ptb_rate = args.ptb_rate

data = Dataset(root='attack_data/', name=args.dataset, seed=42, setting='prognn')
_, features, labels = data.adj, data.features.todense(), data.labels
features = scipy.sparse.csr_matrix(features)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
n_nodes = features.shape[0]
n_class = labels.max() + 1

perturbed_adj = scipy.sparse.load_npz("meta_adj/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate)).todense()
perturbed_adj = torch.FloatTensor(perturbed_adj).to(device)
train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes), idx_to_mask(idx_val, n_nodes), idx_to_mask(idx_test, n_nodes)
train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
logger.info('train nodes:%d' % train_mask.sum())
logger.info('val nodes:%d' % val_mask.sum())
logger.info('test nodes:%d' % test_mask.sum())

# Training parameters
epochs = 200
n_hidden = 16
dropout = 0.5
weight_decay = 5e-4
lr = 1e-2
loss = nn.CrossEntropyLoss()


def train(model, optim, adj, run, embeds, verbose=True):
    best_loss_val = 9999
    best_acc_val = 0
    for epoch in range(epochs):
        model.train()
        logits = model(adj, embeds)
        l = loss(logits[train_mask], labels[train_mask])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, embeds, labels, val_mask)
        val_loss = loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                logger.info("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    torch.save(weights, './save_model/%s_%s_%s.pth' % (args.attack, args.dataset, args.ptb_rate))
    test_acc = evaluate(model, adj, embeds, labels, test_mask)
    logger.info("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return best_acc_val, test_acc



if __name__ == '__main__':
    logger.info(args)
    perturbed_adj_sparse = to_scipy(perturbed_adj)
    logger.info('===start preprocessing the graph===')
    if args.dataset == 'polblogs':
        args.jt = 0
    adj_pre = preprocess_adj(features, perturbed_adj_sparse, logger, threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    _, features = to_tensor(perturbed_adj_sparse, features)
    logger.info('===start getting contrastive embeddings===')
    embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(), adj_delete=adj_delete,
                                    lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta)
    embeds = embeds.squeeze(dim=0)
    val_acc_total = []
    acc_total = []
    embeds = embeds.to('cpu')
    embeds = to_scipy(embeds)

    # prune the perturbed graph by the representations
    adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False, threshold=args.cos)
    embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    features = features.to_dense()
    labels = torch.LongTensor(labels)
    adj_clean = adj_clean.to(device)
    features = features.to(device)
    labels = labels.to(device)
    logger.info('===train ours on perturbed graph===')
    for run in range(10):
        adj_temp = adj_clean.clone()
        # add k new neighbors to each node
        get_reliable_neighbors(adj_temp, embeds, k=args.k, degree_threshold=args.threshold)
        model = GCN(embeds.shape[1], n_hidden, n_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        adj_temp = adj_new_norm(adj_temp, args.alpha)
        best_val_acc, test_acc = train(model, optimizer, adj_temp, run, embeds=embeds, verbose=False)
        val_acc_total.append(best_val_acc)
        acc_total.append(test_acc)
    logger.info('Val Mean Accuracy:%f' % np.mean(val_acc_total))
    logger.info('Val Standard Deviation:%f' % np.std(val_acc_total, ddof=1))
    logger.info('Mean Accuracy:%f' % np.mean(acc_total))
    logger.info('Standard Deviation:%f' % np.std(acc_total, ddof=1))
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
