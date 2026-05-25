"""
Graph Structure Learner for FairUnGSL.

Learns a new adjacency matrix from node embeddings via similarity computation,
KNN sparsification, and FairUnGSL refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from gcn import GraphConvolution


class GraphLearner(nn.Module):
    """
    Graph structure learner using node embedding similarity.

    Pipeline:
    1. Encode nodes with two diagonal GCN heads
    2. Compute pairwise similarity (two-head inner product)
    3. Sparsify via KNN (excluding self-loops)
    4. Symmetrize

    Parameters
    ----------
    n_feat : int
        Number of input features.
    k : int
        Number of nearest neighbors for KNN sparsification.
    """

    def __init__(self, n_feat, k=20):
        super(GraphLearner, self).__init__()

        # Two-head diagonal scaling (following GRCN)
        # Each head applies element-wise learnable scaling to features
        self.weight1 = nn.Parameter(torch.FloatTensor(n_feat))
        self.weight2 = nn.Parameter(torch.FloatTensor(n_feat))
        nn.init.uniform_(self.weight1, 0.0, 1.0)
        nn.init.uniform_(self.weight2, 0.0, 1.0)

        self.k = k

    def compute_similarity(self, x, adj_norm):
        """
        Compute node similarity from features using two-head inner product.

        Head 1: applies weight1 scaling, propagates, then inner product
        Head 2: applies weight2 scaling, propagates, then inner product
        sim = head1 + head2
        """
        # Head 1
        x1 = x * self.weight1  # [N, F] element-wise scaling
        if adj_norm.is_sparse:
            x1 = torch.spmm(adj_norm, x1)
        else:
            x1 = torch.mm(adj_norm, x1)
        x1 = F.relu(x1)

        # Head 2
        x2 = x * self.weight2  # [N, F] element-wise scaling
        if adj_norm.is_sparse:
            x2 = torch.spmm(adj_norm, x2)
        else:
            x2 = torch.mm(adj_norm, x2)
        x2 = F.relu(x2)

        # L2-normalize embeddings before inner product (cosine similarity)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)

        # Two-head cosine similarity (values in [-1, 1])
        sim = torch.mm(x1, x1.t()) + torch.mm(x2, x2.t())

        return sim

    def knn_sparsify(self, sim_matrix):
        """
        KNN sparsification: keep top-k neighbors for each node.
        Self-loops are excluded before KNN selection.
        """
        n = sim_matrix.shape[0]

        # Mask out diagonal (self-loops) before KNN
        sim_matrix = sim_matrix.clone()
        sim_matrix.fill_diagonal_(float('-inf'))

        # Get top-k values and indices for each row
        topk_vals, topk_idx = torch.topk(sim_matrix, self.k, dim=1)

        # Build sparse adjacency
        row_idx = torch.arange(n, device=sim_matrix.device).unsqueeze(1).expand(-1, self.k).flatten()
        col_idx = topk_idx.flatten()
        values = topk_vals.flatten()

        # Ensure non-negative
        values = F.relu(values)

        indices = torch.stack([row_idx, col_idx])
        adj_sparse = torch.sparse_coo_tensor(indices, values, (n, n))

        return adj_sparse

    def symmetrize(self, adj_sparse):
        """Symmetrize: A_sym = (A + A^T) / 2"""
        adj_dense = adj_sparse.to_dense()
        adj_sym = (adj_dense + adj_dense.t()) / 2.0
        return adj_sym

    def forward(self, x, adj_norm):
        """
        Learn a new graph structure.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, F].
        adj_norm : torch.Tensor
            Normalized original adjacency [N, N].

        Returns
        -------
        torch.Tensor
            Learned adjacency matrix [N, N] (dense).
        """
        sim = self.compute_similarity(x, adj_norm)
        adj_sparse = self.knn_sparsify(sim)
        adj_new = self.symmetrize(adj_sparse)
        return adj_new

    def graph_parameters(self):
        """Return graph learner parameters."""
        return [self.weight1, self.weight2]
