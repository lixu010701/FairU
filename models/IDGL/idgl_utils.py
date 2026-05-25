"""
IDGL utility components: graph metric learning, sparsification, and regularization.
Ported from OpenGSL for FairUnGSL project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCosine(nn.Module):
    """Multi-head weighted cosine similarity for graph learning.

    Computes: mean_k( normalize(x * W_k) @ normalize(y * W_k)^T )
    """
    def __init__(self, d_in, num_pers=16):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(num_pers, d_in))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, y=None):
        if y is None:
            y = x
        # x: [N, F], y: [M, F], w: [H, F]
        # expand: [H, 1, F] * [1, N, F] = [H, N, F]
        cx = x.unsqueeze(0) * self.w.unsqueeze(1)
        cy = y.unsqueeze(0) * self.w.unsqueeze(1)
        cx = F.normalize(cx, p=2, dim=-1)
        cy = F.normalize(cy, p=2, dim=-1)
        # [H, N, M] -> mean over heads -> [N, M]
        return torch.matmul(cx, cy.transpose(-1, -2)).mean(0)


class EpsilonNN(nn.Module):
    """Epsilon-neighborhood sparsification: zero out entries below threshold."""
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, adj):
        if self.epsilon is None or self.epsilon == 0:
            return adj
        mask = (adj > self.epsilon).float()
        return adj * mask


class KNN(nn.Module):
    """KNN sparsification: keep only top-K neighbors per node."""
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, adj):
        if self.k is None or self.k <= 0:
            return adj
        topk_val, topk_idx = torch.topk(adj, min(self.k, adj.size(-1)), dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_idx, 1.0)
        return adj * mask


class IDGLGraphLearner(nn.Module):
    """IDGL graph learner: weighted cosine similarity + optional sparsification."""
    def __init__(self, input_size, topk=None, epsilon=None, num_pers=16):
        super().__init__()
        self.metric = WeightedCosine(input_size, num_pers)
        self.enn = EpsilonNN(epsilon) if epsilon else None
        self.knn = KNN(topk) if topk else None

    def forward(self, x, anchor=None):
        """Learn adjacency from node features.

        Args:
            x: [N, F] node features
            anchor: [S, F] anchor features (scalable mode), or None for full mode
        Returns:
            [N, N] or [N, S] similarity matrix
        """
        adj = self.metric(x, y=anchor)
        if self.enn is not None:
            adj = self.enn(adj)
        if self.knn is not None:
            adj = self.knn(adj)
        return adj


def compute_graph_regularization(adj, features, smoothness_ratio=0.0, degree_ratio=0.0, sparsity_ratio=0.0):
    """Compute IDGL graph regularization losses.

    Args:
        adj: learned adjacency [N, N]
        features: node features [N, F]
        smoothness_ratio: weight for feature smoothness loss
        degree_ratio: weight for degree connectivity loss
        sparsity_ratio: weight for sparsity loss
    Returns:
        Scalar loss tensor
    """
    loss = torch.tensor(0.0, device=adj.device)

    if smoothness_ratio > 0:
        # Smoothness: tr(X^T L X) where L = D - A
        degree = adj.sum(dim=-1)
        L = torch.diag(degree) - adj
        loss = loss + smoothness_ratio * torch.trace(features.t() @ L @ features) / adj.shape[0]

    if degree_ratio > 0:
        # Connectivity: encourage non-zero degree
        loss = loss + degree_ratio * (-torch.log(torch.clamp(adj.sum(dim=-1), min=1e-12)).mean())

    if sparsity_ratio > 0:
        # Sparsity: penalize dense adjacency
        loss = loss + sparsity_ratio * torch.norm(adj, p='fro')

    return loss


def diff(X, Y, Z):
    """Normalized difference for convergence check."""
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    return diff_ / torch.clamp(norm_, min=1e-12)


def sample_anchors(node_vec, s):
    """Sample s anchor nodes randomly."""
    idx = torch.randperm(node_vec.size(0), device=node_vec.device)[:s]
    return node_vec[idx], idx
