"""
Standalone GCN implementation for FairUnGSL.
Does not depend on OpenGSL — can be used independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """Simple GCN layer: Z = D^{-1/2} A D^{-1/2} X W"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """
    Two-layer GCN for node classification.

    Parameters
    ----------
    n_feat : int
        Number of input features.
    n_hid : int
        Number of hidden units.
    n_class : int
        Number of output classes.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_feat, n_hid, n_class, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj, return_hidden=False):
        h = F.relu(self.gc1(x, adj))
        self._hidden = h  # Cache for adversarial debiasing (v5)
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.gc2(h, adj)
        if return_hidden:
            return out, h
        return out


def normalize_adj(adj):
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}.
    Handles both dense and sparse tensors.
    Returns same format as input (sparse in → sparse out).
    """
    if not isinstance(adj, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(adj)}")

    if adj.is_sparse:
        return normalize_adj_sparse(adj)

    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def normalize_adj_sparse(adj_sparse):
    """Sparse symmetric normalization: D^{-1/2} A D^{-1/2} (memory efficient)."""
    adj_sparse = adj_sparse.coalesce()
    indices = adj_sparse.indices()
    values = adj_sparse.values()
    n = adj_sparse.shape[0]

    # Compute degree from sparse tensor
    degree = torch.sparse.sum(adj_sparse, dim=1).to_dense()
    d_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))

    # Scale values: val_ij * d_inv_sqrt[i] * d_inv_sqrt[j]
    row, col = indices[0], indices[1]
    new_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]

    return torch.sparse_coo_tensor(indices, new_values, adj_sparse.shape).coalesce()
