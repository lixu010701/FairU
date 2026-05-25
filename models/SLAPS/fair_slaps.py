"""
FairSLAPS: SLAPS + FairUnGSL integration.

SLAPS (NeurIPS 2021, Fatemi et al.): Self-Learning Adjacency via Self-Supervision.
  - Graph learner: 2-layer MLP 闁?inner product 闁?KNN sparsification
  - Core self-supervision: Denoising Autoencoder (DAE) 闁?mask features,
    reconstruct via learned graph propagation
  - Training: two-phase (warmup DAE-only, then +CE)

FairSLAPS modifications:
  1. Replace UnGSL's SparseUnGSL with our FairUnGSLmodule (閿?+ 閿?
  2. Add feature-level debiasing BEFORE SLAPS to avoid DAE-fairness conflict
  3. Expose hidden representation `_hidden` for adversarial/contrastive losses
  4. Support entropy-weighted fairness loss (innovation: use UnGSL's entropy
     as per-node weight for fairness loss 闁?high entropy 闁?boundary/minority node)
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gcn import GCN, normalize_adj
from FairUnGSLmodule import FairUnGSL, FairSparseUnGSL


def symmetrize(adj):
    """Force adjacency matrix to be symmetric."""
    return (adj + adj.T) / 2


def row_normalize(adj):
    """Row-normalize adjacency with self-loops.

    Supports both dense and sparse inputs. Output is dense (for GCN propagation).
    """
    if adj.is_sparse:
        adj = adj.to_dense()
    n = adj.size(0)
    adj_with_loop = adj + torch.eye(n, device=adj.device)
    deg = adj_with_loop.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return adj_with_loop / deg


def knn_sparsify(similarity, k):
    """Keep only top-k entries per row, zero others."""
    values, indices = similarity.topk(k, dim=-1)
    mask = torch.zeros_like(similarity).scatter_(-1, indices, 1.0)
    return similarity * mask


class SLAPSGraphLearner(nn.Module):
    """2-layer MLP + inner product + KNN (SLAPS-style graph learner).

    Initialization: identity weights (so MLP starts as identity),
    then it learns to refine features for better similarity computation.
    """

    def __init__(self, n_feat, hidden_dim, output_dim, k=20, non_linearity='relu'):
        super().__init__()
        self.k = k
        self.non_linearity = non_linearity

        # 2-layer MLP
        self.fc1 = nn.Linear(n_feat, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize: identity-like (SLAPS "mlp_knn_init" when in==out)
        if n_feat == output_dim and hidden_dim == n_feat:
            self.fc1.weight.data = torch.eye(n_feat)
            self.fc2.weight.data = torch.eye(n_feat)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # MLP 闁?embedding
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)

        # L2 normalize for cosine similarity
        h = F.normalize(h, dim=-1, p=2)

        # Inner product similarity
        sim = h @ h.T

        # KNN sparsification
        sim = knn_sparsify(sim, self.k + 1)

        # Non-linearity (ReLU for non-negative edges)
        if self.non_linearity == 'relu':
            sim = F.relu(sim)
        elif self.non_linearity == 'elu':
            sim = F.elu(sim) + 1.0  # shift to non-negative

        return sim


class FairSLAPS(nn.Module):
    """SLAPS graph learner + FairUnGSL (閿?閿? + dual-head GCN (DAE + classifier).

    Forward flow:
        X  闁? [feat debias]  闁? graph_gen (MLP+KNN)  闁? A_raw
                                                         闁?                                                    FairUnGSL (閿?閿? 闁?A_fair
                                                         闁?                                           闁崇懓澶囬弨銏ゅ煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘閳ь剟鍩為埀顒勫煘缁鏁橀柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍?                                           闁?                          闁?                                  GCN_DAE(masked_X, A)         GCN_C(X, A)
                                           闁?                          闁?                                       recon_X                     logits
                                           闁?                          闁?                                      loss_dae                  loss_task + fair
    """

    def __init__(self, n_nodes, n_feat, n_class, entropy, sensitive, conf, device='cuda:0'):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device

        n_hidden = conf.get('n_hidden', 64)
        hidden_adj = conf.get('hidden_adj', n_feat)  # MLP hidden dim for graph learner
        dropout = conf.get('dropout', 0.5)

        # Graph learner (SLAPS style: MLP + KNN)
        self.graph_learner = SLAPSGraphLearner(
            n_feat=n_feat,
            hidden_dim=hidden_adj,
            output_dim=n_feat,  # output same as input for identity init
            k=conf.get('k', 20),
            non_linearity=conf.get('non_linearity', 'relu'),
        )

        # FairUnGSL module (閿?+ 閿? same as other backbones)
        # Use sparse version for large graphs to avoid O(N閾? OOM
        use_sparse = conf.get('use_sparse_fair', n_nodes > 3000)
        fair_conf = {
            'init_value': conf.get('init_value', 0.5),
            'beta': conf.get('beta', 0.1),
            'lambda_init': conf.get('lambda_init', 0.0),
            'delta_eps': conf.get('delta_eps', 0.0),
            'use_fairdrop': conf.get('use_fairdrop', False),
            'fair_mode': conf.get('fair_mode', 'default'),
            'delta_eps_init': conf.get('delta_eps_init', 0.0),
        }
        if use_sparse:
            self.fair_ungsl = FairSparseUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
            print(f"  FairSLAPS using FairSparseUnGSL (n_nodes={n_nodes} > 3000)")
        else:
            self.fair_ungsl = FairUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
            print(f"  FairSLAPS using FairUnGSL dense (n_nodes={n_nodes})")
        self._use_sparse_fair = use_sparse

        # Classifier GCN (main task head)
        self.classifier = GCN(n_feat, n_hidden, n_class, dropout)

        # DAE GCN (reconstruction head) 闁?outputs same dim as features
        dae_hidden = conf.get('dae_hidden', n_hidden)
        self.gcn_dae = GCN(n_feat, dae_hidden, n_feat, dropout)

        # DAE mask ratio
        self.dae_mask_ratio = conf.get('dae_mask_ratio', 0.2)

        # v2: Graph fusion ratio (like GRCN)
        # adj_final = fusion * adj_learned + (1 - fusion) * adj_original
        # fusion=1.0 闁?pure SLAPS (weak on tabular data)
        # fusion=0.0 闁?pure original graph (GCN baseline)
        # fusion=0.5 闁?balanced blend (recommended for tabular)
        self.fusion_ratio = conf.get('fusion_ratio', 0.5)

        # Cache for loss computation
        self._last_recon = None
        self._last_mask = None
        self._adj_orig_norm = None

    def forward(self, x, adj):
        """Forward pass with graph fusion (v2).

        Args:
            x: features [N, F]
            adj: original dataset adjacency [N, N] (sparse or dense)
        Returns:
            output: classifier logits [N, C]
            adj_fair: FairUnGSL-filtered adjacency
            adj_fair: (same, for interface with other backbones)
        """
        # Cache normalized original adj on first forward
        if self._adj_orig_norm is None:
            with torch.no_grad():
                self._adj_orig_norm = row_normalize(adj)

        # 1. Generate graph via MLP + KNN
        adj_raw = self.graph_learner(x)
        adj_raw = symmetrize(adj_raw)

        # 2. Apply FairUnGSL (閿?+ 閿? on learned graph
        adj_fair = self.fair_ungsl(adj_raw)

        # 3. Row-normalize the learned adjacency
        adj_learned_norm = row_normalize(adj_fair)

        # 3b. v2: Fuse with original adjacency (KEY FIX for tabular data)
        # This gives the model access to dataset structure, not just learned graph.
        # Without this, SLAPS on tabular data produces weak representations.
        if self.fusion_ratio < 1.0:
            adj_norm = (self.fusion_ratio * adj_learned_norm
                        + (1.0 - self.fusion_ratio) * self._adj_orig_norm)
        else:
            adj_norm = adj_learned_norm

        # 4. DAE: mask features, reconstruct
        if self.training:
            mask = torch.bernoulli(
                torch.full_like(x, self.dae_mask_ratio))
            x_masked = x * (1 - mask)
            recon_logits = self.gcn_dae(x_masked, adj_norm)
            self._last_recon = recon_logits
            self._last_mask = mask
        else:
            self._last_recon = None
            self._last_mask = None

        # 5. Classification
        output = self.classifier(x, adj_norm)

        return output, adj_fair, adj_fair

    def get_dae_loss(self, features):
        """Compute DAE reconstruction loss on masked positions.

        Uses MSE (for continuous features, like our fairness datasets after
        feature_norm). Original SLAPS uses BCE which only works for binary/
        [0,1] features 闁?unstable for our range.
        """
        if self._last_recon is None or self._last_mask is None:
            return torch.tensor(0.0, device=features.device)
        mask = self._last_mask > 0
        recon = self._last_recon
        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        # MSE for continuous features
        loss = F.mse_loss(recon[mask], features[mask], reduction='mean')
        return loss

    def base_parameters(self):
        return list(self.classifier.parameters()) + list(self.gcn_dae.parameters())

    def graph_parameters(self):
        return list(self.graph_learner.parameters())

    def ungsl_parameters(self):
        return self.fair_ungsl.ungsl_parameters()

    def fairness_parameters(self):
        return self.fair_ungsl.fairness_parameters()
