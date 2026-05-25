"""
FairSublime: SUBLIME (WWW 2022) self-contained port for FairUnGSL.

Phase 1: code-port only. No fairness intervention applied.
Opengsl dependencies are inlined (normalize/symmetry/FGP/AttLearner/MLPLearner).
GCNEncoder is replaced with FairU-GSL's shared models.gcn.GCN in dense mode.
The DGL-backed GCNConv_dgl path is preserved for sparse mode.

Reference: "Towards Unsupervised Deep Graph Structure Learning", Liu et al. WWW 2022.
"""

import os
import sys
import copy
import math
import numpy as np
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn as nn
import torch.nn.functional as F

# DGL is only required in sparse mode; keep import optional.
try:
    import dgl
    import dgl.function as fn
    _HAS_DGL = True
except ImportError:
    dgl = None
    fn = None
    _HAS_DGL = False

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gcn import GCN as FairGCN
from adversarial import compute_adv_loss


EOS = 1e-10


# ------------------------------------------------------------
# Graph-level fairness loss (Phase 2)
# ------------------------------------------------------------

def graph_fair_loss(learned_adj, sens, mask=None, reduction='mean'):
    """Edge-level fairness loss |mean_same - mean_cross|.

    Penalises the absolute difference between the average same-group edge
    weight and the average cross-group edge weight. Supports three input
    forms:
      * Dense [N, N] tensor           闁?materialised mask product (legacy path)
      * torch.sparse COO tensor       闁?operates directly on indices/values
      * DGL graph (sparse SUBLIME)    闁?reads ``edata['w']`` and edges

    The DGL / torch-sparse paths avoid O(N^2) materialisation, so the loss
    works on pokec-scale graphs (66k+ nodes) without OOM.

    Parameters
    ----------
    learned_adj : torch.Tensor or dgl.DGLGraph
        Adjacency representation.
    sens : torch.Tensor
        Binary sensitive attribute vector [N].
    mask : unused
        Kept for API symmetry with FairnessLoss.
    reduction : unused
        Kept for API symmetry.

    Returns
    -------
    torch.Tensor
        Scalar loss on the same device as the input adjacency.
    """
    # --- DGL graph path: work directly on edges, preserve grad on edata['w'] ---
    if _HAS_DGL and isinstance(learned_adj, dgl.DGLGraph):
        device = learned_adj.device
        s = sens.float().to(device)
        rows_, cols_ = learned_adj.edges()
        values = learned_adj.edata['w']
        cross = (s[rows_.long()] - s[cols_.long()]).abs()
        same = 1.0 - cross
        edge_kept = (values > 0).float()
        cross_w = (values * cross * edge_kept).sum()
        same_w = (values * same * edge_kept).sum()
        n_cross = (cross * edge_kept).sum().clamp(min=1)
        n_same = (same * edge_kept).sum().clamp(min=1)
        return ((same_w / n_same) - (cross_w / n_cross)).abs()

    device = learned_adj.device if torch.is_tensor(learned_adj) else sens.device
    s = sens.float().to(device)

    # --- torch.sparse path: iterate only over explicit edges ---
    if torch.is_tensor(learned_adj) and learned_adj.is_sparse:
        adj_c = learned_adj.coalesce()
        indices = adj_c.indices()
        values = adj_c.values()
        rows_, cols_ = indices[0], indices[1]
        cross = (s[rows_] - s[cols_]).abs()
        same = 1.0 - cross
        edge_kept = (values > 0).float()
        cross_w = (values * cross * edge_kept).sum()
        same_w = (values * same * edge_kept).sum()
        n_cross = (cross * edge_kept).sum().clamp(min=1)
        n_same = (same * edge_kept).sum().clamp(min=1)
        return ((same_w / n_same) - (cross_w / n_cross)).abs()

    # --- Dense path (unchanged) ---
    edge_mask = (learned_adj > 0).float()
    cross_mask = (s.unsqueeze(0) - s.unsqueeze(1)).abs()
    same_mask = 1.0 - cross_mask

    cross_edges = learned_adj * cross_mask * edge_mask
    same_edges = learned_adj * same_mask * edge_mask

    n_cross = (cross_mask * edge_mask).sum().clamp(min=1)
    n_same = (same_mask * edge_mask).sum().clamp(min=1)

    mean_cross = cross_edges.sum() / n_cross
    mean_same = same_edges.sum() / n_same

    return (mean_same - mean_cross).abs()


# ------------------------------------------------------------
# DGL <-> torch.sparse conversions and feature masking utils
# ------------------------------------------------------------

def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape, device=features.device)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask, samples


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    return torch.sparse_coo_tensor(indices, values, size=(dgl_graph.num_nodes(),
                                                          dgl_graph.num_nodes()))


def torch_sparse_to_dgl_graph(torch_sparse_mx, device='cuda:0'):
    if not _HAS_DGL:
        raise RuntimeError("DGL not available; sparse SUBLIME path requires dgl.")
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    g = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device=device)
    g.edata['w'] = values.detach().to(device)
    return g


# ------------------------------------------------------------
# Inlined opengsl.functional: normalize + symmetry
# ------------------------------------------------------------

def _normalize_dense(adj, add_loop=False):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0], device=device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    return r_mat_inv @ adj_loop @ r_mat_inv


def _normalize_sparse(adj, add_loop=False):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size())


def normalize(adj, add_loop=False):
    """Symmetric normalization D^{-1/2} A D^{-1/2} (dense or sparse)."""
    if adj.is_sparse:
        return _normalize_sparse(adj, add_loop)
    return _normalize_dense(adj, add_loop)


def symmetry(adj, i=2):
    if adj.is_sparse:
        n = adj.shape[0]
        adj_t = torch.sparse_coo_tensor(adj.indices()[[1, 0]], adj.values(), size=(n, n))
        return (adj_t + adj).coalesce() / i
    return (adj.t() + adj) / i


# ------------------------------------------------------------
# Sparse GCN conv over a DGL graph
# ------------------------------------------------------------

class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


# ------------------------------------------------------------
# Similarity metrics (inlined from opengsl.metric)
# ------------------------------------------------------------

class WeightedCosine(nn.Module):
    """Weighted cosine similarity with multi-head projections."""

    def __init__(self, d_in, num_pers=1, weighted=True, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.w = None
        if weighted:
            self.w = nn.Parameter(torch.FloatTensor(num_pers, d_in))
            self.reset_parameters()

    def reset_parameters(self):
        if self.w is not None:
            nn.init.xavier_uniform_(self.w)

    def forward(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        context_x = x.unsqueeze(0)
        context_y = y.unsqueeze(0)
        if self.w is not None:
            expand_weight_tensor = self.w.unsqueeze(1)
            context_x = context_x * expand_weight_tensor
            context_y = context_y * expand_weight_tensor
        if self.normalize:
            context_x = F.normalize(context_x, p=2, dim=-1)
            context_y = F.normalize(context_y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.transpose(-1, -2)).mean(0)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask
        return adj


class CosineMetric(nn.Module):
    def forward(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        context_x = F.normalize(x, p=2, dim=-1)
        context_y = F.normalize(y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask
        return adj


# ------------------------------------------------------------
# FGP: fully-parameterized NxN adjacency initialized from kNN
# ------------------------------------------------------------

class FGP(nn.Module):
    def __init__(self, n, nonlinear=None, init_adj=None):
        super().__init__()
        self.Adj = nn.Parameter(torch.FloatTensor(n, n))
        self.nonlinear = lambda adj: F.elu(adj) + 1
        if nonlinear is not None:
            self.nonlinear = eval(nonlinear)
        if init_adj is not None:
            self.Adj.data.copy_(init_adj)

    def reset_parameters(self, features, k, metric, i):
        adj = kneighbors_graph(features, k, metric=metric)
        adj = np.array(adj.todense(), dtype=np.float32)
        adj += np.eye(adj.shape[0])
        adj = adj * i - i
        self.Adj.data.copy_(torch.tensor(adj))

    def forward(self, x=None):
        return self.nonlinear(self.Adj)


# ------------------------------------------------------------
# Attentive / MLP encoders for learners
# ------------------------------------------------------------

class AttentiveLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.w = nn.Parameter(torch.ones(d))

    def reset_parameters(self):
        self.w = nn.Parameter(self.w.new_ones(self.d))

    def forward(self, x):
        return x @ torch.diag(self.w)


class AttentiveEncoder(nn.Module):
    def __init__(self, n_layers, d, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList([AttentiveLayer(d) for _ in range(n_layers)])
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (len(self.layers) - 1):
                x = self.activation(x)
        return x


class MLPEncoderSUB(nn.Module):
    """Identity-initialized MLP encoder for MLPLearner (SUBLIME style)."""

    def __init__(self, n_feat, n_hidden, n_class, n_layers, dropout=0.0,
                 use_bn=False, activation='relu'):
        super().__init__()
        self.n_feat = n_feat
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu

        if n_layers == 1:
            self.lins.append(nn.Linear(n_feat, n_class))
        else:
            self.lins.append(nn.Linear(n_feat, n_hidden))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(n_hidden))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(n_hidden, n_hidden))
                if use_bn:
                    self.bns.append(nn.BatchNorm1d(n_hidden))
            self.lins.append(nn.Linear(n_hidden, n_class))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def param_init(self):
        # SUBLIME: identity init so learner starts as pass-through
        for layer in self.lins:
            layer.weight = nn.Parameter(torch.eye(self.n_feat))

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn and i < len(self.bns):
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


# ------------------------------------------------------------
# Post-processing: KNN + non-linearity
# ------------------------------------------------------------

def _knn_dense(adj, K, self_loop=True, set_value=None):
    device = adj.device
    values, indices = adj.topk(k=int(K), dim=-1)
    mask = torch.zeros(adj.shape, device=device)
    mask[torch.arange(adj.shape[0], device=device).view(-1, 1), indices] = 1.
    if not self_loop:
        diag = torch.arange(adj.shape[0], device=device).view(-1, 1)
        mask[diag, diag] = 0
    mask.requires_grad = False
    new_adj = adj * mask
    if set_value is not None:
        new_adj[new_adj.nonzero(as_tuple=True)] = set_value
    return new_adj


def _apply_non_linearity(adj, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(adj * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(adj)
    elif non_linearity == 'none':
        return adj
    raise KeyError(f'Unsupported non_linearity: {non_linearity}')


def _knn_fast(X, k, b):
    """Sparse KNN used by DGL-based learners (adapted from opengsl)."""
    device = X.device
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1), device=device)
    rows = torch.zeros(X.shape[0] * (k + 1), device=device)
    cols = torch.zeros(X.shape[0] * (k + 1), device=device)
    norm_row = torch.zeros(X.shape[0], device=device)
    norm_col = torch.zeros(X.shape[0], device=device)
    while index < X.shape[0]:
        end = min(index + b, X.shape[0])
        sub_tensor = X[index:end]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):end * (k + 1)] = vals.reshape(-1)
        cols[index * (k + 1):end * (k + 1)] = inds.reshape(-1)
        rows[index * (k + 1):end * (k + 1)] = (
            torch.arange(index, end, device=device).view(-1, 1).repeat(1, k + 1).reshape(-1)
        )
        norm_row[index:end] = vals.sum(dim=1)
        norm_col.index_add_(-1, inds.reshape(-1), vals.reshape(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values = values * (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


# ------------------------------------------------------------
# Graph learners
# ------------------------------------------------------------

class AttLearner(nn.Module):
    """Attention-based similarity learner (SUBLIME Att variant)."""

    def __init__(self, n_layers, isize, k, i, sparse, act):
        super().__init__()
        self.encoder = AttentiveEncoder(n_layers, isize, activation=act)
        self.metric = WeightedCosine(isize, num_pers=1, weighted=False, normalize=True)
        self.k = k
        self.i = i
        self.sparse = sparse
        self.non_linearity = 'relu'

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, x):
        if self.sparse:
            embeddings = self.encoder(x)
            rows, cols, values = _knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = _apply_non_linearity(values_, self.non_linearity, self.i)
            g = dgl.graph((rows_, cols_), num_nodes=x.shape[0], device=x.device)
            g.edata['w'] = values_
            return g
        x = self.encoder(x)
        adj = self.metric(x)
        adj = _knn_dense(adj, self.k + 1, self_loop=True)
        adj = _apply_non_linearity(adj, self.non_linearity, self.i)
        return adj


class MLPLearner(nn.Module):
    """MLP-based similarity learner (SUBLIME MLP variant)."""

    def __init__(self, n_layers, isize, k, i, sparse, act):
        super().__init__()
        self.encoder = MLPEncoderSUB(isize, isize, isize, n_layers,
                                     dropout=0.0, use_bn=False, activation=act)
        self.encoder.param_init()
        self.metric = CosineMetric()
        self.k = k
        self.i = i
        self.sparse = sparse
        self.non_linearity = 'relu'

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, x):
        if self.sparse:
            embeddings = self.encoder(x)
            rows, cols, values = _knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = _apply_non_linearity(values_, self.non_linearity, self.i)
            g = dgl.graph((rows_, cols_), num_nodes=x.shape[0], device=x.device)
            g.edata['w'] = values_
            return g
        x = self.encoder(x)
        adj = self.metric(x)
        adj = _knn_dense(adj, self.k + 1, self_loop=True)
        adj = _apply_non_linearity(adj, self.non_linearity, self.i)
        return adj


# ------------------------------------------------------------
# Encoders / classifiers using FairU-GSL's shared GCN in dense mode
# ------------------------------------------------------------

class _DenseGCNStack(nn.Module):
    """Thin dense GCN stack sharing logic with FairU-GSL's models.gcn.GCN.

    The existing FairGCN is 2-layer. For configurations with n_layers > 2, we
    fall back to stacking linear + adj-mul layers directly here (still pure
    torch, still dense). For n_layers == 2 we reuse FairGCN.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        if n_layers == 2:
            self._use_fair = True
            self.gcn = FairGCN(in_dim, hidden_dim, out_dim, dropout=dropout)
        else:
            self._use_fair = False
            dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(dims[i], dims[i + 1])) for i in range(n_layers)
            ])
            self.biases = nn.ParameterList([
                nn.Parameter(torch.zeros(dims[i + 1])) for i in range(n_layers)
            ])
            for w in self.weights:
                stdv = 1.0 / math.sqrt(w.size(1))
                w.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self._use_fair:
            return self.gcn(x, adj)
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if adj.is_sparse:
                h = torch.sparse.mm(adj, h)
            else:
                h = adj @ h
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphEncoder(nn.Module):
    """SUBLIME GraphEncoder: GCN trunk + MLP projection head."""

    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim,
                 dropout, sparse):
        super().__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
            self.model = None
        else:
            self.model = _DenseGCNStack(in_dim, hidden_dim, emb_dim,
                                        n_layers=nlayers, dropout=dropout)
        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x, Adj_):
        if self.sparse:
            h = x
            for conv in self.gnn_encoder_layers[:-1]:
                h = conv(h, Adj_)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gnn_encoder_layers[-1](h, Adj_)
        else:
            h = self.model(x, Adj_)
        z = self.proj_head(h)
        return z, h


class GCL(nn.Module):
    """Graph contrastive learner: two-view encoder + NT-Xent loss."""

    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim,
                 dropout, dropout_adj, sparse):
        super().__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim,
                                    proj_dim, dropout, sparse)
        self.dropout_adj = dropout_adj
        self.sparse = sparse

    def forward(self, x, Adj_, branch=None):
        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj,
                                       training=self.training)
        else:
            Adj = F.dropout(Adj_, p=self.dropout_adj, training=self.training)
        z, emb = self.encoder(x, Adj)
        return z, emb

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (
            torch.einsum('i,j->ij', x_abs, x_aug_abs) + EOS)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + EOS)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + EOS)
        loss_0 = -torch.log(loss_0 + EOS).mean()
        loss_1 = -torch.log(loss_1 + EOS).mean()
        return (loss_0 + loss_1) / 2.0


class GCN_SUB(nn.Module):
    """Stage-2 classifier operating on the learned adjacency."""

    def __init__(self, nfeat, nhid, nclass, n_layers=2, dropout=0.5,
                 dropout_adj=0.5, sparse=0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.sparse = sparse
        self.dropout_adj_p = dropout_adj
        self.dropout = dropout
        if sparse:
            self.layers.append(GCNConv_dgl(nfeat, nhid))
            for _ in range(n_layers - 2):
                self.layers.append(GCNConv_dgl(nhid, nhid))
            self.layers.append(GCNConv_dgl(nhid, nclass))
            self.model = None
        else:
            self.model = _DenseGCNStack(nfeat, nhid, nclass,
                                        n_layers=n_layers, dropout=dropout)

    def forward(self, x, Adj):
        if self.sparse:
            Adj = copy.deepcopy(Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p,
                                       training=self.training)
            h = x
            for conv in self.layers[:-1]:
                h = conv(h, Adj)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[-1](h, Adj)
            return h.squeeze(1)
        Adj = F.dropout(Adj, p=self.dropout_adj_p, training=self.training)
        return self.model(x, Adj)


# ------------------------------------------------------------
# FairSublimeModel: wrapper exposing Phase-1 components
# ------------------------------------------------------------

class FairSublimeModel(nn.Module):
    """Wrapper holding GCL + graph_learner + classifier.

    In Phase 1 the ``ungsl_module`` argument is accepted but unused. Phase 2
    will wire it in via :meth:`apply_fair_adjustment`.
    """

    def __init__(self, n_feat, n_hidden, n_embed, n_proj, n_class,
                 n_layers=2, n_layers_cls=2, dropout=0.5, dropout_adj=0.5,
                 dropout_cls=0.5, dropedge_cls=0.25, sparse=0,
                 type_learner='mlp', k=30, sim_function='cosine',
                 activation_learner='relu', n_nodes=None, features_cpu=None,
                 device='cuda:0', ungsl_module=None):
        super().__init__()
        self.sparse = bool(sparse)
        self.type_learner = type_learner
        self.device = device
        self.n_nodes = n_nodes
        self.ungsl_module = ungsl_module  # Phase 2 placeholder

        self.graph_learner = self._build_graph_learner(
            type_learner=type_learner, n_feat=n_feat, k=k,
            sim_function=sim_function, activation_learner=activation_learner,
            n_nodes=n_nodes, features_cpu=features_cpu, sparse=self.sparse,
        )

        self.model = GCL(
            nlayers=n_layers, in_dim=n_feat, hidden_dim=n_hidden,
            emb_dim=n_embed, proj_dim=n_proj,
            dropout=dropout, dropout_adj=dropout_adj, sparse=self.sparse,
        )

        self.classifier = GCN_SUB(
            nfeat=n_feat, nhid=n_hidden, nclass=n_class,
            n_layers=n_layers_cls, dropout=dropout_cls,
            dropout_adj=dropedge_cls, sparse=self.sparse,
        )

    @staticmethod
    def _build_graph_learner(type_learner, n_feat, k, sim_function,
                             activation_learner, n_nodes, features_cpu, sparse):
        t = type_learner.lower()
        if t == 'fgp':
            learner = FGP(n_nodes)
            if features_cpu is None:
                raise ValueError("FGP learner requires features_cpu for init.")
            learner.reset_parameters(features_cpu, k, sim_function, 6)
            return learner
        if t == 'mlp':
            return MLPLearner(n_layers=2, isize=n_feat, k=k, i=6,
                              sparse=sparse, act=activation_learner)
        if t == 'att':
            return AttLearner(n_layers=2, isize=n_feat, k=k, i=6,
                              sparse=sparse, act=activation_learner)
        raise ValueError(f"Unknown type_learner: {type_learner}")

    # ------------------------------------------------------------
    # Phase 2: FairUnGSL integration hook.
    # ------------------------------------------------------------
    def set_ungsl(self, ungsl_module):
        """Attach a FairUnGSL / FairSparseUnGSL module for fair adjustment."""
        self.ungsl_module = ungsl_module

    def apply_fair_adjustment(self, learned_adj, sens, entropy_vector=None, mode='D'):
        """Apply FairUnGSL refinement to the learned adjacency.

        Parameters
        ----------
        learned_adj : torch.Tensor or dgl.DGLGraph
            Dense [N, N] adjacency (dense mode) or DGL graph (sparse mode)
            from the graph learner.
        sens : torch.Tensor
            Binary sensitive attribute vector.
        entropy_vector : unused
            Entropy is already baked into ``self.ungsl_module``.
        mode : unused
            Kept for API symmetry with other backbones.

        Returns
        -------
        torch.Tensor or dgl.DGLGraph
            Refined adjacency (matches the input type).
        """
        if self.ungsl_module is None:
            return learned_adj
        if self.sparse:
            # Sparse path: DGL graph -> torch.sparse (grad-preserving) ->
            # FairSparseUnGSL -> DGL graph.
            # We cannot reuse the module-level `dgl_graph_to_torch_sparse`
            # because it calls `.cpu().detach()` on edata['w'], which breaks
            # backprop into `ungsl_module.thresholds`.
            rows_, cols_ = learned_adj.edges()
            values = learned_adj.edata['w']
            n = learned_adj.num_nodes()
            indices = torch.stack([rows_.long(), cols_.long()], dim=0)
            adj_sp = torch.sparse_coo_tensor(
                indices, values, size=(n, n)
            ).coalesce()
            adj_sp_refined = self.ungsl_module(adj_sp)
            # Rebuild DGL graph preserving grad on values.
            adj_sp_refined = adj_sp_refined.coalesce()
            new_idx = adj_sp_refined.indices()
            new_vals = adj_sp_refined.values()
            g = dgl.graph(
                (new_idx[0], new_idx[1]),
                num_nodes=n, device=self.device,
            )
            g.edata['w'] = new_vals
            return g
        return self.ungsl_module(learned_adj)

    # Optional: expose parameter groups in the same style as other backbones
    def cl_parameters(self):
        return list(self.model.parameters())

    def learner_parameters(self):
        return list(self.graph_learner.parameters())

    def classifier_parameters(self):
        return list(self.classifier.parameters())


# ------------------------------------------------------------
# Phase 2: Stage-2 fairness-aware classifier head
# ------------------------------------------------------------

class FairSublimeStage2Head(nn.Module):
    """Wraps a stage-2 GCN classifier with a combined fairness loss.

    Combines:
      * Cross-entropy task loss (with optional class weighting).
      * Statistical-parity loss (``FairnessLoss.compute_sp_loss``).
      * Equal-opportunity loss (``FairnessLoss.compute_eo_loss``).
      * Structure-level fairness loss (``FairnessLoss.compute_structure_loss``).
      * Adversarial sensitive-prediction loss via GRL (optional).

    The total loss is returned together with a dict of scalar sub-losses for
    logging.
    """

    def __init__(self, gcn_classifier, discriminator=None, fair_loss_fn=None):
        super().__init__()
        self.classifier = gcn_classifier
        self.discriminator = discriminator
        self.fair_loss_fn = fair_loss_fn

    def compute_loss(self, logits, labels, sens, train_mask, epoch,
                     hidden=None, learned_adj=None, adv_weight=0.0,
                     class_weights=None):
        device = logits.device
        mask = train_mask

        ce = F.cross_entropy(logits[mask], labels[mask], weight=class_weights)

        sp_val = 0.0
        eo_val = 0.0
        struct_val = 0.0
        adv_val = 0.0

        fair_term = torch.tensor(0.0, device=device)
        if self.fair_loss_fn is not None:
            alpha = self.fair_loss_fn.get_alpha(epoch)

            sp_loss = self.fair_loss_fn.compute_sp_loss(logits, sens, mask)
            eo_loss = self.fair_loss_fn.compute_eo_loss(logits, sens, labels, mask)

            if learned_adj is not None:
                # FairnessLoss.compute_structure_loss expects a tensor (dense or torch.sparse).
                # SUBLIME sparse mode passes a DGL graph; convert to torch.sparse first.
                if hasattr(learned_adj, 'edata') and hasattr(learned_adj, 'edges'):
                    _src, _dst = learned_adj.edges()
                    _vals = learned_adj.edata['w']
                    _idx = torch.stack([_src, _dst], dim=0)
                    _N = learned_adj.num_nodes()
                    _adj_for_struct = torch.sparse_coo_tensor(
                        _idx, _vals, (_N, _N), device=device
                    ).coalesce()
                    struct_loss = self.fair_loss_fn.compute_structure_loss(_adj_for_struct, sens)
                else:
                    struct_loss = self.fair_loss_fn.compute_structure_loss(learned_adj, sens)
            else:
                struct_loss = torch.tensor(0.0, device=device)

            sp_weighted = self.fair_loss_fn.alpha_sp * sp_loss
            eo_weighted = self.fair_loss_fn.alpha_eo * eo_loss
            struct_weighted = self.fair_loss_fn.alpha_struct * struct_loss

            fair_term = alpha * (sp_weighted + eo_weighted + struct_weighted)

            sp_val = float(sp_loss.detach().item()) if torch.is_tensor(sp_loss) else float(sp_loss)
            eo_val = float(eo_loss.detach().item()) if torch.is_tensor(eo_loss) else float(eo_loss)
            struct_val = float(struct_loss.detach().item()) if torch.is_tensor(struct_loss) else float(struct_loss)
        else:
            alpha = 1.0

        total = ce + fair_term

        if self.discriminator is not None and hidden is not None and adv_weight > 0:
            adv_loss = compute_adv_loss(self.discriminator, hidden, sens, mask,
                                        alpha=1.0)
            total = total + alpha * adv_weight * adv_loss
            adv_val = float(adv_loss.detach().item())

        return total, {
            'ce': float(ce.detach().item()),
            'sp': sp_val,
            'eo': eo_val,
            'struct': struct_val,
            'adv': adv_val,
        }
