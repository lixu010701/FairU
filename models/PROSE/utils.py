"""
Utility functions for PROSE backbone.
Adapted from the PROSE reference implementation with minimal changes for FairU-GSL.
"""

import sys
import os
_BACKBONE_DIR = os.path.dirname(__file__)
if _BACKBONE_DIR not in sys.path:
    sys.path.insert(0, _BACKBONE_DIR)

import torch
import torch.nn.functional as F
import numpy as np

VERY_SMALL_NUMBER = 1e-12


def get_feat_mask(features, mask_rate):
    """Generate random feature mask for data augmentation."""
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape).to(features.device)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask, samples


def split_batch(node_idxs, batch_size):
    """Split node indices into batches."""
    batches = []
    for i in range(0, len(node_idxs), batch_size):
        batches.append(node_idxs[i:i + batch_size])
    return batches


def torch_sparse_eye(n):
    """Create sparse identity matrix."""
    indices = torch.arange(n).unsqueeze(0).repeat(2, 1)
    values = torch.ones(n)
    return torch.sparse_coo_tensor(indices, values, (n, n))


def normalize(adj, style='sym', sparse=True):
    """Normalize adjacency matrix (symmetric or row normalization)."""
    if sparse and adj.is_sparse:
        adj = adj.coalesce()
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        if style == 'sym':
            d_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
            indices = adj.indices()
            values = adj.values() * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
            return torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()
        else:  # row normalize
            d_inv = torch.where(degree > 0, degree.pow(-1), torch.zeros_like(degree))
            indices = adj.indices()
            values = adj.values() * d_inv[indices[0]]
            return torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()
    else:
        if adj.is_sparse:
            adj = adj.to_dense()
        degree = adj.sum(dim=1)
        if style == 'sym':
            d_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
            return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        else:
            d_inv = torch.where(degree > 0, degree.pow(-1), torch.zeros_like(degree))
            return d_inv.unsqueeze(1) * adj


def extract_subgraph(adj, idx):
    """Extract subgraph given node indices."""
    if adj.is_sparse:
        adj = adj.to_dense()
    return adj[idx][:, idx]


def compute_anchor_adj(node_anchor_adj):
    """Convert node-anchor adjacency to approximate node-node adjacency."""
    anchor_norm = node_anchor_adj / torch.clamp(
        torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER
    )
    node_norm = node_anchor_adj / torch.clamp(
        torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER
    )
    return torch.matmul(anchor_norm, node_norm.transpose(-1, -2))


def add_graph_degree_loss(adj, weight):
    """Graph degree regularization loss."""
    return -weight * torch.log(adj.sum(1) + 1e-12).mean()
