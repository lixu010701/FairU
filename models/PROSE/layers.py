"""
AnchorGCNLayer for PROSE backbone.
Adapted from the PROSE reference implementation with minimal changes for FairU-GSL.
"""

import sys
import os
_BACKBONE_DIR = os.path.dirname(__file__)
if _BACKBONE_DIR not in sys.path:
    sys.path.insert(0, _BACKBONE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = 1e-10
VERY_SMALL_NUMBER = 1e-12


class AnchorGCNLayer(nn.Module):
    """GCN layer supporting both standard sparse and anchor-based message passing."""

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=False, batch_norm=False):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            # Column-normalize: N * anchor_num
            node_norm = node_anchor_adj / torch.clamp(
                torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER
            )
            # Row-normalize: N * anchor_num
            anchor_norm = node_anchor_adj / torch.clamp(
                torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER
            )
            output = torch.matmul(
                anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support)
            )
        else:
            if adj.is_sparse:
                output = torch.sparse.mm(adj, support)
            else:
                output = torch.mm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self._compute_bn(output)

        return output

    def _compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())
