# Graph Adversarial Diffusion Convolution 

This repository contains an implementation of ["Graph Adversarial Diffusion Convolution"](https://openreview.net/pdf?id=ICvWruTEDH), which is a general and principled framework for defense against feature noise and graph sturcture attacks.  



## Defense Performance against Non-adaptive Graph Structure Attack

```bash
cd Non-Adaptive_Adversarial_Attack 
```

### 1. Generate disrupted graph structure
While we provide the disrupted adjacency matrix in the meta_adj folder, you can also use the following commands to generated the disrupted adjacency matrix.
```bash
python setup.py --dataset cora --ptb_rate 0.25  
```

### 2. Baseline Evaluation

### GCN
```bash
python gcn.py --dataset cora --ptb_rate 0.25 
# We can replace dataset and ptb_rate  
```

### GAT
```bash
python gat.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### APPNP
```bash
python appnp.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### SSGC
```bash
python ssgc.py --epochs 100 --lr 0.2 --weight_decay 1e-5 --alpha 0.05 --degree 16 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python ssgc.py --epochs 150 --lr 0.2 --weight_decay 1e-4 --alpha 0.05 --degree 16 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate  

python ssgc.py --epochs 100 --lr 0.2 --weight_decay 2e-5 --alpha 0.05 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

### NAGphormer
```bash
python main.py --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python main.py --batch_size 2000 --dropout 0.3 --hidden_dim 512  --hops 7  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate  

python main.py --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 7  --n_heads 8 --n_layers 1 --pe_dim 15 --peak_lr 0.001  --weight_decay=1e-05 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

### Robust GCN
```bash
python rgcn.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### GCN Jaccard
```bash
python gcn_jaccard.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### GCN SVD
```bash
python gcn_svd.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### Pro-GNN
```bash
python prognn.py --alpha 5e-4 --beta 1.5 --gamma 1 --lambda_ 0.001 --lr 5e-4 --epoch 1000 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python prognn.py --alpha 5e-4 --beta 1.5 --gamma 1 --lambda_ 0.0001 --lr 5e-4 --epoch 1000 --dataset citeseer --ptb_rate 0.25
# We can replace ptb_rate  

python prognn.py --alpha 0.3 --beta 2.5 --gamma 1 --lambda_ 0.001 --lr 1e-2 --epoch 100 --inner_steps 30 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

### GNNGuard
```bash
python main.py --modelname GCN --GNNGuard True --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

### Elastic GNN
```bash
cd code
python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset Cora-adv --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset CiteSeer-adv --ptb_rate 0.25 
# We can replace ptb_rate 

python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset PubMed-adv --ptb_rate 0.25  
# We can replace ptb_rate 
```

### HANG-quad
```bash
python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 8 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 12 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 6 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset pubmed --ptb_rate 0.25   
# We can replace ptb_rate 
```

### STABLE
```bash
python main.py --alpha 0.6 --beta 2 --k 7  --jt 0.03 --cos 0.25 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --alpha 0.1 --beta 2 --k 5  --jt 0.03 --cos 0.1 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --alpha 0.1 --beta 2 --k 5  --jt 0.03 --cos 0.1 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate 
```

### GCN-GARNET
```bash
python main.py --device 0 --backbone gcn --dataset cora --attack meta --ptb_rate 0.25 --perturbed
# We can replace dataset and ptb_rate  
```

### EvenNet
```bash
python main.py --runs 100 --dataset cora --ptb_rate 0.25 --alpha 0.9  
# We can replace ptb_rate 

python main.py --runs 100 --dataset citeseer --ptb_rate 0.25 --alpha 0.9    
# We can replace ptb_rate 

python main.py --runs 100 --dataset pubmed --ptb_rate 0.25 --alpha 0.5 
python main.py --runs 100 --dataset pubmed --ptb_rate 0.5 --alpha 0.9
python main.py --runs 100 --dataset pubmed --ptb_rate 0.75 --alpha 0.9
```

### 3. GADC Evaluation

### GADC
```bash
python gadc.py --degree 6 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.25  
python gadc.py --degree 3 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.5  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.75  
python gadc.py --degree 6 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.25  
python gadc.py --degree 3 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.5  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.75  
python gadc.py --degree 2 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-5 --hidden 32 --dataset pubmed --ptb_rate 0.25  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-5 --hidden 32 --dataset pubmed --ptb_rate 0.5 
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-4 --hidden 32 --dataset pubmed --ptb_rate 0.75 
```


## Defense Performance against Adaptive Graph Structure Attack

```bash
cd Adaptive_Adversarial_Attack
python gcn.py --dataset cora
python gnnguard.py --dataset cora
python sftgdc.py --dataset cora
python svd_gcn.py --dataset cora
python gadc.py --dataset cora

# We can replace dataset
```


## Denoising Performance against Feature Noise
```bash
cd Denoising
```

### 1. Baseline Evaluation

### MLP
```bash
python mlp.py --noise_type gaussian --runs 100 --dataset cora --noise_level 0.1
# We can replace dataset and noise_level  

python mlp.py --noise_type flip --runs 100 --dataset cora --ber 0.1
# We can replace dataset and ber  

python mlp.py --noise_type gaussian --runs 10 --dataset coauthor-cs --noise_level 0.1
# We can replace dataset and noise_level  

python mlp_products.py --noise_level 0.1
# We can replace noise_level 
```

### GCN
```bash
python gcn_denoising.py --noise_type gaussian --runs 100 --dataset cora --noise_level 0.1
# We can replace dataset and noise_level  

python gcn_denoising.py --noise_type flip --runs 100 --dataset cora --ber 0.1
# We can replace dataset and ber  

python gcn_denoising.py --noise_type gaussian --runs 10 --dataset coauthor-cs --noise_level 0.1
# We can replace dataset and noise_level  

python gcn_products.py --noise_level 0.1
# We can replace noise_level 
```

### GAT
```bash
python gat.py --noise_type gaussian --runs 100 --lr 0.005 --weight_decay 5e-4 --dataset cora --noise_level 0.1
python gat.py --noise_type gaussian --runs 100 --lr 0.005 --weight_decay 5e-4 --dataset citeseer --noise_level 0.1
python gat.py --noise_type gaussian --runs 100 --lr 0.01 --weight_decay 0.001 --heads_2 8 --dataset pubmed --noise_level 0.01
# We can replace noise_level  

python gat.py --noise_type flip --runs 100 --lr 0.005 --weight_decay 5e-4 --dataset cora --noise_level 0.1
python gat.py --noise_type flip --runs 100 --lr 0.005 --weight_decay 5e-4 --dataset citeseer --noise_level 0.1
python gat.py --noise_type flip --runs 100 --lr 0.01 --weight_decay 0.001 --heads_2 8 --dataset pubmed --noise_level 0.1
# We can replace ber  

python gat.py --noise_type gaussian --runs 10 --lr 0.005 --weight_decay 5e-4 --dataset cora --ber 0.1
python gat.py --noise_type gaussian --runs 10 --lr 0.005 --weight_decay 5e-4 --dataset citeseer --ber 0.1
python gat.py --noise_type gaussian --runs 10 --lr 0.01 --weight_decay 0.001 --heads_2 8 --dataset pubmed --ber 0.1
# We can replace noise_level  
```

### GLP
```bash
python glp.py --noise_type gaussian --runs 100 --dataset cora --noise_level 0.1
# We can replace dataset and noise_level  

python glp.py --noise_type flip --runs 100 --dataset cora --ber 0.1
# We can replace dataset and ber  

python glp.py --noise_type gaussian --runs 10 --dataset coauthor-cs --noise_level 0.1
# We can replace dataset and noise_level  
```

### SSGC
```bash
python ssgc.py --noise_type gaussian --runs 100 --dataset cora --weight_decay 1e-5 --epochs 100 --noise_level 0.1
python ssgc.py --noise_type gaussian --runs 100 --dataset citeseer --weight_decay 1e-4 --epochs 150 --noise_level 0.1
python ssgc.py --noise_type gaussian --runs 100 --dataset pubmed --weight_decay 2e-5 --epochs 100 --noise_level 0.01
# We can replace noise_level  

python ssgc.py --noise_type flip --runs 100 --dataset cora --weight_decay 1e-5 --epochs 100 --ber 0.1
python ssgc.py --noise_type flip --runs 100 --dataset citeseer --weight_decay 1e-4 --epochs 150 --ber 0.1
python ssgc.py --noise_type flip --runs 100 --dataset pubmed --weight_decay 2e-5 --epochs 100 --ber 0.1
# We can replace ber  

python ssgc.py --noise_type gaussian --runs 10 --dataset coauthor-cs --weight_decay 1e-5 --epochs 100 --noise_level 0.1
# We can replace dataset and noise_level  

python ssgc_products.py --noise_level 0.1
# We can replace noise_level 
```

### IRLS
```bash
python irls.py --noise_type=gaussian --data=cora --mlp_bef=1 --mlp_aft=0 --dropout=0.8 --prop_step=8 --alp=1 --lam=1 --inp_dropout=0.8 --lr=0.3 --weight_decay=5e-5 --runs=100 --seed=42 --noise_level=0.1
python irls.py --noise_type=gaussian --data=citeseer --mlp_bef=1 --mlp_aft=0 --prop_step=16 --lr=0.1 --num_epoch=500 --inp_dropout=0.5 --lam=1 --alp=1 --weight_decay=0.001 --runs=100 --seed=42 --noise_level=0.1
python irls.py --noise_type=gaussian --data=pubmed --mlp_bef=1 --mlp_aft=0 --prop_step=40 --lr=0.5 --num_epoch=500 --inp_dropout=0.8 --lam=1 --alp=1 --weight_decay=0.0005 --runs=100 --seed=42 --noise_level=0.01
# We can replace noise_level  

python irls.py --noise_type=flip --data=cora --mlp_bef=1 --mlp_aft=0 --dropout=0.8 --prop_step=8 --alp=1 --lam=1 --inp_dropout=0.8 --lr=0.3 --weight_decay=5e-5 --runs=100 --seed=42 --ber=0.1
python irls.py --noise_type=flip --data=citeseer --mlp_bef=1 --mlp_aft=0 --prop_step=16 --lr=0.1 --num_epoch=500 --inp_dropout=0.5 --lam=1 --alp=1 --weight_decay=0.001 --runs=100 --seed=42 --ber=0.1
python irls.py --noise_type=flip --data=pubmed --mlp_bef=1 --mlp_aft=0 --prop_step=40 --lr=0.5 --num_epoch=500 --inp_dropout=0.8 --lam=1 --alp=1 --weight_decay=0.0005 --runs=100 --seed=42 --ber=0.1
# We can replace ber  

python irls.py --noise_type=gaussian --data=coauthor-cs --mlp_bef=1 --mlp_aft=0 --prop_step=40 --lr=0.5 --num_epoch=500 --inp_dropout=0.8 --lam=1 --alp=1 --weight_decay=0.0005 --runs=10 --seed=42 --noise_level=0.1
python irls.py --noise_type=gaussian --data=coauthor-cs --mlp_bef=1 --mlp_aft=0 --prop_step=40 --lr=0.5 --num_epoch=500 --inp_dropout=0.8 --lam=1 --alp=1 --weight_decay=0.0005 --runs=10 --seed=42 --noise_level=0.1
# We can replace dataset and noise_level  
```

### AirGNN
```bash
python airgnn.py --dataset cora --noise_level 0.1
```

### APPNP
```bash
python appnp.py --noise_type gaussian --runs 100 --dataset cora --noise_level 0.1
# We can replace dataset and noise_level  

python appnp.py --noise_type flip --runs 100 --dataset cora --ber 0.1
# We can replace dataset and ber  

python appnp.py --noise_type gaussian --runs 10 --dataset coauthor-cs --noise_level 0.1
# We can replace dataset and noise_level  
```

### GADC
```bash
python gadc.py --noise_type gaussian --runs 100 --dataset cora --noise_level 0.1
# We can replace dataset and noise_level  

python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-5 --dataset cora --ber 0.1
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-5 --dataset cora --ber 0.2
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-1 --dataset cora --ber 0.4
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-5 --dataset citeseer --ber 0.1
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-5 --dataset citeseer --ber 0.2
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-5 --dataset citeseer --ber 0.4
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-1 --dataset pubmed --ber 0.1
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-1 --dataset pubmed --ber 0.2
python gadc.py --noise_type flip --runs 100 --degree 32 --lam 64 --epsilon 1e-1 --dataset pubmed --ber 0.4

python gadc.py --noise_type gaussian --dataset coauthor-cs --runs 10 --noise_level 0.1 --lam 1 --degree 16 --lr 0.2 --epochs 1000 --weight_decay 1e-7
python gadc.py --noise_type gaussian --dataset coauthor-cs --runs 10 --noise_level 1 --lam 128 --degree 16 --lr 0.2 --epochs 1000 --weight_decay 1e-7
# We can replace dataset

python gadc_products.py --noise_level 0.1 --lam 32
python gadc_products.py --noise_level 1.0 --lam 256
```


## Improving Performance in Heterophilic Graphs
```bash
cd Heterophilic_Graph
```

```bash
python3 gadc.py --dataset cornell --epsilon 1.0
# We can replace dataset and epsilon  

python3 gadc_link.py --dataset cornell --epsilon 1.0
# We can replace dataset and epsilon  
```

## Citation
```
@inproceedings{liu2024graph,
  title={Graph Adversarial Diffusion Convolution},
  author={Liu, Songtao and Chen, Jinghui and Fu, Tianfan and Lin, Lu and Zitnik, Marinka and Wu, Dinghao},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```
