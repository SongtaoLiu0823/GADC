from .base_classes import ODEblock
import torch
from .utils import get_rw_adj, gcn_norm_fill_val
import numpy as np
#from torch_geometric.utils import to_edge_index
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.typing import SparseTensor

def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    r"""Converts a :class:`torch.sparse.Tensor` or a
    :class:`torch_sparse.SparseTensor` to edge indices and edge attributes.

    Args:
        adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_torch_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        adj = adj._coalesced_(True)
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if torch_geometric.typing.WITH_PT112 and adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")

class ConstantODEblock_PLOT(ODEblock):
  def __init__(self, odefunc,  opt,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_PLOT, self).__init__(odefunc,  opt,  device, t)

    self.aug_dim = 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt,  device)
    # if opt['data_norm'] == 'rw':
    #   edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                                                fill_value=opt['self_loop_weight'],
    #                                                                num_nodes=data.num_nodes,
    #                                                                dtype=data.x.dtype)
    # else:
    #   edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
    #                                        fill_value=opt['self_loop_weight'],
    #                                        num_nodes=data.num_nodes,
    #                                        dtype=data.x.dtype)
    #
    #
    #
    # self.odefunc.edge_index = edge_index.to(device)
    # self.odefunc.edge_weight = edge_weight.to(device)
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if (opt['method'] == 'symplectic_euler' or opt['method'] == 'leapfrog'):
      from odeint_geometric import odeint
    elif opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    self.device = device
    self.opt = opt

  def forward(self, x,adj):
    self.t = torch.linspace(0, 10, 100)
    t = self.t.type_as(x)

    integrator = self.train_integrator if self.training else self.test_integrator
    
    # reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    # state = (x,) + reg_states if self.training and self.nreg > 0 else x

    func = self.odefunc
    state = x

    # self.edge_index = adj[0]
    # self.edge_attr = adj[1]
    # self.edge_index = adj._indices()
    # self.edge_attr = adj._values()
    # write if adj is tuple:


    if isinstance(adj, tuple):
      self.edge_index = adj[0]
      self.edge_attr = adj[1]
    else:
      self.edge_index, self.edge_attr = to_edge_index(adj)
    # if self.opt['data_norm'] == 'rw':
    #   edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
    #                                        fill_value=self.opt['self_loop_weight'],
    #                                        num_nodes=self.opt['num_nodes'],
    #                                        dtype=x.dtype)
    # else:
    #   edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
    #                                               fill_value=self.opt['self_loop_weight'],
    #                                               num_nodes=self.opt['num_nodes'],
    #                                               dtype=x.dtype)
    edge_index = self.edge_index
    edge_weight = self.edge_attr
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    # print("adj: ", adj.shape)
    # print("x: ", x)
    # print("t: ", t)
    # print("edge_index shape ", self.odefunc.edge_index.shape)
    # print("edge_weight shape ", self.odefunc.edge_weight.shape)
    # print("edge_index: ", self.odefunc.edge_index)
    # print("edge_weight: ", self.odefunc.edge_weight)
    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      # state_dt = integrator(
      #   func, state, t,
      #   method=self.opt['method'],
      #   options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
      #   atol=self.atol,
      #   rtol=self.rtol)
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size']),
        atol=self.atol,
        rtol=self.rtol)

    # if self.training and self.nreg > 0:
    #   z = state_dt[0][1]
    #   reg_states = tuple( st[1] for st in state_dt[1:] )
    #   return z, reg_states
    # else:
    #   z = state_dt[1]
    # if self.training:
    #   z = state_dt
    # else:
    #   z = state_dt[-1]
    #z = state_dt[1]
    # save the torch state_dt to numpy file
    # if self.training:
    # print("saving state_dt to numpy file")
    state_save = []
    state_save.append(x.detach().cpu().numpy())
    # print("x: ", x.shape)
    for i in range(len(state_dt)):

      state_save.append(state_dt[i].detach().cpu().numpy())
      # print("state_dt: ", state_dt[i].shape)
    np.save('state_save_clean'+ self.opt['function']+'.npy', state_save)


    z = state_dt[-1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
