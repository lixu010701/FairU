"""
FairPROGNN: Pro-GNN + FairUnGSL integration.

Pro-GNN learns a clean adjacency matrix via a free-parameter [N,N] matrix
with Frobenius (graph fidelity), L1 (sparsity), nuclear norm (low-rank),
and feature smoothness regularization.

This implementation uses differentiable approximations of L1 and nuclear
norm so that Pro-GNN fits into FairU-GSL's single-backward 4-optimizer
training framework, avoiding Pro-GNN's original alternating PGD scheme.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gcn import GCN, normalize_adj
from FairUnGSLmodule import FairUnGSL, FairSparseUnGSL


class FGP(nn.Module):
    """Feature Graph Prediction 闁?learnable adjacency matrix.

    A free-parameter [N, N] matrix initialized from the original adjacency.
    Identity nonlinearity (raw values), clamped to [0, 1] after each step.
    """

    def __init__(self, n_nodes):
        super(FGP, self).__init__()
        self.Adj = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))

    def init_from_adj(self, adj_dense):
        """Initialize Adj from the original adjacency (dense)."""
        self.Adj.data.copy_(adj_dense)

    def forward(self):
        """Return clamped adjacency estimate."""
        return torch.clamp(self.Adj, min=0, max=1)


def symmetrize(adj):
    """Symmetrize: (A + A^T) / 2."""
    return (adj + adj.t()) / 2


def smoothness_loss(x, adj):
    """Feature smoothness regularizer: tr(X^T L X) where L = I - D^{-1/2}AD^{-1/2}.

    Encourages connected nodes to have similar features.
    """
    n = x.shape[0]
    adj_sym = symmetrize(adj)
    degree = adj_sym.sum(1)
    d_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    norm_adj = d_inv_sqrt.unsqueeze(1) * adj_sym * d_inv_sqrt.unsqueeze(0)
    L = torch.eye(n, device=x.device) - norm_adj
    return torch.trace(x.t() @ L @ x)


class FairPROGNN(nn.Module):
    """
    Pro-GNN + FairUnGSL model for fair graph structure learning.

    Architecture:
    1. FGP: free-parameter adjacency matrix (initialized from original adj)
    2. FairUnGSL: fairness-aware uncertainty masking on estimated adj
    3. GCN classifier: node classification on refined graph

    Regularization losses (computed via `compute_graph_reg_loss()`):
    - Frobenius: ||Adj_est - Adj_orig||_F  (graph fidelity)
    - L1: ||Adj_est||_1                    (sparsity)
    - Nuclear: sum(svdvals(Adj_est))       (low-rank)
    - Smoothness: tr(X^T L X)             (feature smoothness)
    """

    def __init__(self, n_nodes, n_feat, n_class, entropy, sensitive, conf, device='cuda:0'):
        super(FairPROGNN, self).__init__()

        self.n_nodes = n_nodes

        # --- GCN classifier ---
        self.classifier = GCN(
            n_feat=n_feat,
            n_hid=conf.get('n_hidden', 64),
            n_class=n_class,
            dropout=conf.get('dropout', 0.5),
        )

        # --- FGP adjacency estimator ---
        self.fgp = FGP(n_nodes)

        # Fair mode
        self.fair_mode = conf.get('fair_mode', 'default')

        # --- FairUnGSL module ---
        fair_conf = {
            'init_value': conf.get('init_value', 0.5),
            'beta': conf.get('beta', 0.1),
            'lambda_init': conf.get('lambda_init', 0.0),
            'delta_eps': conf.get('delta_eps', 0.0),
            'use_fairdrop': conf.get('use_fairdrop', False),
            'fair_mode': self.fair_mode,
            'delta_eps_init': conf.get('delta_eps_init', 0.0),
        }
        # Auto-select sparse vs dense FairUnGSL based on graph size
        # For large graphs (>5000), dense FairUnGSL buffers ([N,N] * 3) would OOM
        use_sparse = conf.get('use_sparse', n_nodes > 5000)
        if use_sparse:
            print(f"  Using FairSparseUnGSL (n_nodes={n_nodes} > 5000)")
            self.fair_ungsl = FairSparseUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
        else:
            print(f"  Using FairUnGSL (dense, n_nodes={n_nodes})")
            self.fair_ungsl = FairUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
        self._use_sparse_fair = use_sparse

        # Whether to symmetrize the learned adj
        self.symmetric = conf.get('prognn_symmetric', True)

        # Regularization weights (read from config, used by compute_graph_reg_loss)
        self.reg_fro = conf.get('prognn_gamma', 1.0)      # Frobenius (graph fidelity)
        self.reg_l1 = conf.get('prognn_alpha', 5e-4)       # L1 sparsity
        self.reg_nuclear = conf.get('prognn_beta', 1.5)     # Nuclear norm (low-rank)
        self.reg_smooth = conf.get('prognn_lambda', 0.001)  # Feature smoothness

        # Cached original adj (dense, set on first forward)
        self._adj_orig_dense = None
        self._features_cache = None
        self._adj_est_cache = None  # cached adj_est from forward, reused by reg loss
        self._adj_norm_cache = None  # interface compat with FairGRCN (contrastive path)

    def _ensure_caches(self, x, adj):
        """Cache original adj (dense) and initialize FGP on first call."""
        if self._adj_orig_dense is None:
            with torch.no_grad():
                if adj.is_sparse:
                    self._adj_orig_dense = adj.to_dense().detach()
                else:
                    self._adj_orig_dense = adj.detach()
                # Initialize FGP adjacency from original adj
                self.fgp.init_from_adj(self._adj_orig_dense)
        self._features_cache = x.detach()

    def forward(self, x, adj):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, F].
        adj : torch.Tensor
            Original adjacency matrix (fixed, does not change).

        Returns
        -------
        output : torch.Tensor
            Classification logits [N, C].
        adj_new : torch.Tensor
            Adjacency after FairUnGSL refinement.
        adj_final : torch.Tensor
            Final adjacency used for classification (same as adj_new for PROGNN).
        """
        self._ensure_caches(x, adj)

        # 1. Get estimated adjacency from FGP (clamped to [0,1])
        adj_est = self.fgp()

        # 2. Symmetrize if configured
        if self.symmetric:
            adj_est = symmetrize(adj_est)

        # Cache for compute_graph_reg_loss() to reuse (same computation graph)
        self._adj_est_cache = adj_est

        # 3. Apply FairUnGSL refinement
        if self.fair_mode == 'F':
            # Mode F: 閿?only on adj_est, then detach + 閿?閿?for classification
            adj_psi = self.fair_ungsl.apply_psi_only(adj_est)
            adj_new = self.fair_ungsl(adj_psi.detach())
        elif self.fair_mode == 'D':
            adj_psi = self.fair_ungsl.apply_psi_only(adj_est)
            adj_new = self.fair_ungsl.apply_phi_only(adj_psi)
        else:
            adj_new = self.fair_ungsl(adj_est)

        # 4. Normalize and classify
        if self._use_sparse_fair:
            adj_final_norm = normalize_adj(adj_est)
            output = self.classifier(x, adj_final_norm)
        else:
            adj_final_norm = normalize_adj(adj_new)
            output = self.classifier(x, adj_final_norm)

        return output, adj_new, adj_new

    def compute_graph_reg_loss(self):
        """Compute PROGNN-specific graph regularization losses.

        Called from train.py and added to the total loss.
        Must be called AFTER forward() so caches are populated.

        Returns
        -------
        loss_reg : torch.Tensor
            Weighted sum of Frobenius + L1 + nuclear + smoothness losses.
        """
        # Reuse adj_est cached during forward() 闁?same computation graph
        adj_est = self._adj_est_cache

        loss = torch.tensor(0.0, device=adj_est.device)

        # Frobenius: closeness to original adjacency
        if self.reg_fro > 0:
            loss_fro = torch.norm(adj_est - self._adj_orig_dense, p='fro')
            loss = loss + self.reg_fro * loss_fro

        # L1: sparsity (differentiable)
        if self.reg_l1 > 0:
            loss_l1 = torch.norm(adj_est, p=1)
            loss = loss + self.reg_l1 * loss_l1

        # Nuclear norm: low-rank (differentiable via svdvals)
        # GPU SVD can fail to converge (produces NaN with only a warning, not exception)
        if self.reg_nuclear > 0:
            try:
                svdvals = torch.linalg.svdvals(adj_est)
                if not torch.isnan(svdvals).any():
                    loss_nuclear = svdvals.sum()
                    loss = loss + self.reg_nuclear * loss_nuclear
            except RuntimeError:
                pass

        # Feature smoothness (skip for large graphs 闁?smoothness_loss creates [N,N] intermediates)
        if self.reg_smooth > 0 and self._features_cache is not None and self.n_nodes <= 5000:
            loss_smooth = smoothness_loss(self._features_cache, adj_est)
            loss = loss + self.reg_smooth * loss_smooth

        return loss

    def base_parameters(self):
        """GCN classifier parameters."""
        return list(self.classifier.parameters())

    def graph_parameters(self):
        """Graph estimator parameters (FGP.Adj)."""
        return [self.fgp.Adj]

    def ungsl_parameters(self):
        """UnGSL threshold parameters."""
        return self.fair_ungsl.ungsl_parameters()

    def fairness_parameters(self):
        """Fairness strength parameters."""
        return self.fair_ungsl.fairness_parameters()
