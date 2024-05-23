import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate) :
        super(MLP, self).__init__()
        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers-1+int(init_activate) # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(self.norm_cnt)])


        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)

    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x) # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        return x 


    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i , layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1: # do not activate in the last layer
                x = self.activate(x)

        return x


class IRLS(nn.Module):
    def __init__(self, input_d, output_d, hidden_d, prop_step, num_mlp_before, num_mlp_after, norm, alp, lam, dropout, inp_dropout):
        super(IRLS, self).__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.hidden_d = hidden_d,
        self.prop_step = prop_step
        self.num_mlp_before = num_mlp_before
        self.num_mlp_after = num_mlp_after
        self.norm = norm
        self.dropout =dropout
        self.inp_dropout = inp_dropout
        self.prop_step = prop_step
        self.alp = alp
        self.lam = lam

        # if only one layer, then no hidden size
        self.size_bef_unf = self.hidden_d
        self.size_aft_unf = self.hidden_d
        if self.num_mlp_before == 0:
            self.size_aft_unf = self.input_d  # as the input  of mlp_aft
        if self.num_mlp_after == 0:
            self.size_bef_unf = self.output_d # as the output of mlp_bef
        
        if isinstance(self.hidden_d, tuple):
            self.hidden_d = self.hidden_d[0]

        self.mlp_bef = MLP(input_d=self.input_d, 
                           hidden_d=self.hidden_d, 
                           output_d=self.size_bef_unf, 
                           num_layers=self.num_mlp_before, 
                           dropout=self.dropout, 
                           norm=self.norm, 
                           init_activate=False)
        
        self.mlp_aft = MLP(input_d=self.size_bef_unf, 
                           hidden_d=self.hidden_d, 
                           output_d=self.output_d, 
                           num_layers=self.num_mlp_after, 
                           dropout=self.dropout, 
                           norm=self.norm, 
                           init_activate = (self.num_mlp_before > 0) and (self.num_mlp_after > 0))

    
    def forward(self, x, sem_adj, norm_diag):
        if self.inp_dropout > 0:
            x = F.dropout(x, self.inp_dropout, training = self.training)
        x = self.mlp_bef(x)
        Y = x

        for _ in range(self.prop_step):
            Y = (1 - self.alp) * Y + self.alp * self.lam * torch.spmm(sem_adj, Y) + self.alp * torch.spmm(norm_diag, x)

        x = self.mlp_aft(Y)

        return x

