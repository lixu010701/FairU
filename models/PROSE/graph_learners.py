"""
Stage_GNN_learner for PROSE backbone.
Adapted from the PROSE reference implementation with minimal changes for FairU-GSL.
BetaReLU and AnchorUnGSL are NOT included; those are replaced by FairU-GSL.
"""

import sys
import os
_BACKBONE_DIR = os.path.dirname(__file__)
if _BACKBONE_DIR not in sys.path:
    sys.path.insert(0, _BACKBONE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import AnchorGCNLayer
    from .utils import extract_subgraph
except ImportError:
    from layers import AnchorGCNLayer
    from utils import extract_subgraph


class Stage_GNN_learner(nn.Module):
    """
    Progressive graph structure learner with anchor-based adjacency.

    Uses multi-head cosine similarity to compute node-anchor adjacency,
    epsilon-neighborhood filtering, progressive graph pruning (stage_ks),
    and upsampling stages.
    """

    def __init__(self, isize, osize, head_num, sparse, ks, anchor_adj_fusion_ratio, epsilon):
        super(Stage_GNN_learner, self).__init__()

        self.weight_tensor1 = torch.Tensor(head_num, isize)
        self.weight_tensor1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor1))

        self.weight_tensor2 = torch.Tensor(head_num, osize)
        self.weight_tensor2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor2))

        self.sparse = sparse
        self.anchor_adj_fusion_ratio = anchor_adj_fusion_ratio
        self.epsilon = epsilon

        # Stage module for progressive pooling
        self.ks = ks
        self.l_n = len(self.ks)

        if self.l_n > 0:
            # Score layer for graph pruning
            self.score_layer = AnchorGCNLayer(isize, 1)

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        """Apply epsilon-neighborhood filtering to attention matrix."""
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def knn_anchor_node(self, context, anchors, weight_tensor, k=100, b=500):
        """
        Compute node-anchor adjacency via multi-head cosine similarity.

        Args:
            context: Node features [N, D]
            anchors: Anchor node features [anchor_num, D]
            weight_tensor: Multi-head weight [head_num, D]
            k: Number of nearest neighbors (unused, kept for API compatibility)
            b: Batch size (unused, kept for API compatibility)

        Returns:
            attention: Node-anchor attention matrix [N, anchor_num]
        """
        expand_weight_tensor = weight_tensor.unsqueeze(1)  # [head_num, 1, D]
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        # context: [N, D] -> context_fc: [head_num, N, D]
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)

        # anchors: [anchor_num, D] -> anchors_fc: [head_num, anchor_num, D]
        anchors_fc = anchors.unsqueeze(0) * expand_weight_tensor
        anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)

        # Multi-head cosine similarity averaged across heads
        attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)

        return attention

    def forward_anchor(self, features, ori_adj, anchor_nodes_idx, encoder, fusion_ratio):
        """
        Forward pass for anchor-based graph learning with progressive pruning.

        Args:
            features: Node features [N, D]
            ori_adj: Original adjacency matrix (sparse or dense)
            anchor_nodes_idx: Indices of anchor nodes
            encoder: GNN encoder for computing embeddings during upsampling
            fusion_ratio: Ratio for fusing anchor-based and standard message passing

        Returns:
            node_anchor_adj: Learned node-anchor adjacency matrix [N, anchor_num]
        """
        # Step 1: Initial node-anchor adjacency via cosine similarity
        node_anchor_adj = self.knn_anchor_node(
            features, features[anchor_nodes_idx], self.weight_tensor1
        )
        node_anchor_adj = self.build_epsilon_neighbourhood(node_anchor_adj, self.epsilon, 0)

        if self.l_n > 0:
            indices_list = []
            n_node = features.shape[0]
            pre_idx = torch.arange(0, n_node).long()

            embeddings_ = features
            adj_ = ori_adj

            # Progressive graph pruning (downsampling stages)
            for i in range(self.l_n):
                y = torch.sigmoid(
                    self.score_layer(embeddings_[pre_idx, :], adj_).squeeze()
                )

                score, idx = torch.topk(y, max(2, int(self.ks[i] * adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]

                # Map to global node indices
                pre_idx = pre_idx.to(features.device)
                pre_idx = pre_idx[new_idx]

                indices_list.append(pre_idx)

                adj_ = extract_subgraph(adj_, new_idx)

                # Apply score mask to embeddings
                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(
                    embeddings_,
                    torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1 - mask_score, -1).detach()
                )

            # Upsampling stages (reverse order)
            for j in reversed(range(self.l_n)):
                node_anchor_vec = encoder(embeddings_, node_anchor_adj, True, False)
                node_vec = encoder(embeddings_, ori_adj, False, False)
                node_vec = fusion_ratio * node_anchor_vec + (1 - fusion_ratio) * node_vec

                new_node_anchor_adj = self.knn_anchor_node(
                    node_vec, node_vec[anchor_nodes_idx], self.weight_tensor2
                )
                new_node_anchor_adj = self.build_epsilon_neighbourhood(
                    new_node_anchor_adj, self.epsilon, 0
                )

                # Fuse old and new node-anchor adjacency based on pruned nodes
                mask = torch.ones(n_node).to(features.device)
                mask[indices_list[j]] = self.anchor_adj_fusion_ratio
                node_anchor_adj = (
                    torch.mul(node_anchor_adj, torch.unsqueeze(mask, -1))
                    + torch.mul(new_node_anchor_adj, torch.unsqueeze(1 - mask, -1).detach())
                )

        return node_anchor_adj
