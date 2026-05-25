"""
FairGRCN: GRCN + FairUnGSL integration.

Combines graph structure learning (GRCN-style) with fairness-aware
uncertainty-based edge refinement.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both package-level and direct script execution
# Ensure project root is on sys.path so shared modules can be imported.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Shared modules at project root (models/gcn.py and FairUnGSLmodule.py)
from gcn import GCN, normalize_adj
from FairUnGSLmodule import FairUnGSL, FairSparseUnGSL

# Local to this backbone (same directory)
try:
    from .graph_learner import GraphLearner
except ImportError:
    from graph_learner import GraphLearner


class FairGRCN(nn.Module):
    """
    GRCN + FairUnGSL model for fair graph structure learning.

    Architecture:
    1. GraphLearner: learn new adjacency from node embeddings
    2. FairUnGSL: refine adjacency with fairness-aware uncertainty masking
    3. Graph fusion: interpolate learned and original adjacency
    4. GCN classifier: node classification on fused graph

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_feat : int
        Number of input features.
    n_class : int
        Number of output classes.
    entropy : torch.Tensor
        Pre-computed node entropy [N].
    sensitive : torch.Tensor
        Binary sensitive attributes [N].
    conf : dict
        Configuration dictionary.
    device : str
        Device string.
    """

    def __init__(self, n_nodes, n_feat, n_class, entropy, sensitive, conf, device='cuda:0'):
        super(FairGRCN, self).__init__()

        # --- GCN classifier ---
        self.classifier = GCN(
            n_feat=n_feat,
            n_hid=conf.get('n_hidden', 64),
            n_class=n_class,
            dropout=conf.get('dropout', 0.5),
        )

        # --- Graph Learner ---
        self.graph_learner = GraphLearner(
            n_feat=n_feat,
            k=conf.get('k', 20),
        )

        # --- FairUnGSL module ---
        # Auto-select sparse vs dense based on graph size (OOM prevention)
        fair_conf = {
            'init_value': conf.get('init_value', 0.5),
            'beta': conf.get('beta', 0.1),
            'lambda_init': conf.get('lambda_init', 0.0),
            # v4: group-conditional 閽?offset + FairDrop-style hard drop
            'delta_eps': conf.get('delta_eps', 0.0),
            'use_fairdrop': conf.get('use_fairdrop', False),
            'use_phi': conf.get('use_phi', True),
        }
        use_sparse = conf.get('use_sparse', n_nodes > 5000)
        if use_sparse:
            print(f"  Using FairSparseUnGSL (n_nodes={n_nodes} > 5000)")
            self.fair_ungsl = FairSparseUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
        else:
            print(f"  Using FairUnGSL (dense, n_nodes={n_nodes})")
            self.fair_ungsl = FairUnGSL(n_nodes, entropy, sensitive, fair_conf, device)

        # Fusion weight
        self.fusion_ratio = conf.get('fusion_ratio', 0.5)

        # Skip graph learning for large graphs (memory efficiency)
        self.skip_graph_learning = conf.get('skip_graph_learning', n_nodes > 5000)
        if self.skip_graph_learning:
            print(f"  Skipping graph learning (n_nodes={n_nodes} > 5000), applying FairUnGSL directly to original adj")

        # Cached normalized adjacency (set on first forward call)
        self._adj_norm_cache = None
        self._adj_dense_cache = None

    def forward(self, x, adj):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, F].
        adj : torch.Tensor
            Original adjacency matrix [N, N] (fixed, does not change).

        Returns
        -------
        output : torch.Tensor
            Classification logits [N, C].
        adj_new : torch.Tensor
            Adjacency after FairUnGSL refinement.
        adj_final : torch.Tensor
            Final adjacency used for classification.
        """
        # Cache normalized original adj (computed once, stays on GPU)
        if self._adj_norm_cache is None:
            with torch.no_grad():
                if self.skip_graph_learning:
                    # Large graph: use sparse normalization to avoid OOM
                    from gcn import normalize_adj_sparse
                    adj_sp = adj if adj.is_sparse else adj.to_sparse()
                    self._adj_norm_cache = normalize_adj_sparse(adj_sp).detach()
                else:
                    self._adj_norm_cache = normalize_adj(adj).detach()

        if self.skip_graph_learning:
            # Large graph mode: GCN on cached normalized adj, FairUnGSL for fairness loss
            output = self.classifier(x, self._adj_norm_cache)
            # FairUnGSL produces sparse adj for fairness loss (no dense allocation)
            adj_new_sparse = self.fair_ungsl(adj)
            return output, adj_new_sparse, adj_new_sparse
        else:
            # Full mode: graph learning + FairUnGSL + fusion
            if self._adj_dense_cache is None:
                with torch.no_grad():
                    self._adj_dense_cache = (adj.to_dense() if adj.is_sparse else adj).detach()

            adj_learned = self.graph_learner(x, self._adj_norm_cache)
            adj_new = self.fair_ungsl(adj_learned)
            adj_final = self.fusion_ratio * adj_new + (1 - self.fusion_ratio) * self._adj_dense_cache
            adj_final_norm = normalize_adj(adj_final)
            output = self.classifier(x, adj_final_norm)
            return output, adj_new, adj_final

    def base_parameters(self):
        """GCN classifier parameters."""
        return list(self.classifier.parameters())

    def graph_parameters(self):
        """Graph learner parameters."""
        return self.graph_learner.graph_parameters()

    def ungsl_parameters(self):
        """UnGSL threshold parameters."""
        return self.fair_ungsl.ungsl_parameters()

    def fairness_parameters(self):
        """Fairness strength parameters."""
        return self.fair_ungsl.fairness_parameters()
