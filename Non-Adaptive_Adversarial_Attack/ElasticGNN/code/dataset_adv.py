import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
import scipy.sparse as sp

from deeprobust.graph.data import Dataset as DeepRobust_Dataset
from deeprobust.graph.data import PrePtbDataset as DeepRobust_PrePtbDataset
from torch_geometric.data import Data

from util import index_to_mask, mask_to_index
from dataset import get_transform


def get_dataset(args, split, sparse=True):

    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None

    assert args.dataset in ["Cora-adv", "CiteSeer-adv", "PubMed-adv", "Polblogs-adv"], 'dataset not supported'
    if args.dataset == "Cora-adv":
        dataset = get_adv_dataset('cora', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    
    elif args.dataset == "CiteSeer-adv":
        dataset = get_adv_dataset('citeseer', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    
    elif args.dataset == "PubMed-adv":
        dataset = get_adv_dataset('pubmed', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)

    elif args.dataset == "Polblogs-adv":
        dataset = get_adv_dataset('polblogs', args.normalize_features, transform=transform, ptb_rate=args.ptb_rate, args=args)
    data = dataset.data

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test']  = mask_to_index(data.test_mask)
    
    return dataset, data, split_idx


def get_adv_dataset(name, normalize_features=False, transform=None, ptb_rate=0.05, args=None):
    transform = get_transform(normalize_features, transform)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'attack_data/')
    dataset = DeepRobust_Dataset(root=path, name=name, setting='prognn', require_mask=True, seed=42)
    dataset.x = torch.FloatTensor(dataset.features.todense())
    dataset.y = torch.LongTensor(dataset.labels)
    dataset.num_classes = dataset.y.max().item() + 1

    if ptb_rate > 0:
        meat_adj_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'meta_adj/')
        perturbed_adj = sp.load_npz("%s/%s_meta_adj_%s.npz" % (meat_adj_path, name, ptb_rate))
        edge_index = torch.LongTensor(perturbed_adj.nonzero())
    else:
        edge_index = torch.LongTensor(dataset.adj.nonzero())
    data = Data(x=dataset.x, edge_index=edge_index, y=dataset.y)
    
    clean_edge_index = torch.LongTensor(dataset.adj.nonzero())
    clean_data = Data(x=dataset.x, edge_index=clean_edge_index, y=dataset.y)
    
    data.train_mask = torch.tensor(dataset.train_mask)
    data.val_mask   = torch.tensor(dataset.val_mask)
    data.test_mask  = torch.tensor(dataset.test_mask)

    dataset.data = transform(data)
    dataset.clean_data = transform(clean_data)
    dataset.data.clean_adj = dataset.clean_data.adj_t
    return dataset
