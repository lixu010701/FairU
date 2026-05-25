"""
FairIDGL: IDGL + FairUnGSL integration.

IDGL (Iterative Deep Graph Learning) learns graph structure via weighted cosine
similarity with iterative refinement 闁?first learns from raw features, then
refines using GCN hidden representations until convergence.

Two modes:
- Normal mode (small graphs, <=3000 nodes): full [N, N] adjacency + FairUnGSL dense
- Scalable mode (large graphs, >3000 nodes): [N, anchor_num] + FairAnchorUnGSL
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_BACKBONE_DIR = os.path.dirname(__file__)
if _BACKBONE_DIR not in sys.path:
    sys.path.insert(0, _BACKBONE_DIR)

from gcn import GCN, normalize_adj
from FairUnGSLmodule import FairUnGSL, FairSparseUnGSL

# Import FairAnchorUnGSL from PROSE backbone
_PROSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PROSE'))
if _PROSE_DIR not in sys.path:
    sys.path.insert(0, _PROSE_DIR)
from fair_prose import FairAnchorUnGSL

from idgl_utils import (IDGLGraphLearner, compute_graph_regularization,
                        diff, sample_anchors)


class AnchorGCNLayer(nn.Module):
    """GCN layer supporting anchor-based message passing for scalable IDGL."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x, adj, anchor_mp=False):
        support = torch.mm(x, self.weight)
        if anchor_mp:
            # Node-anchor-node message passing
            node_norm = adj / torch.clamp(adj.sum(dim=-2, keepdim=True), min=1e-12)
            anchor_norm = adj / torch.clamp(adj.sum(dim=-1, keepdim=True), min=1e-12)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            if adj.is_sparse:
                output = torch.spmm(adj, support)
            else:
                output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class ScalableGCN(nn.Module):
    """GCN with anchor message passing + skip connection for scalable IDGL."""

    def __init__(self, n_feat, n_hid, n_class, dropout=0.5):
        super().__init__()
        self.layer1 = AnchorGCNLayer(n_feat, n_hid)
        self.layer2 = AnchorGCNLayer(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, init_adj, cur_anchor_adj, graph_skip_conn,
                first=True, first_agg=None, update_ratio=None, first_anchor_adj=None):
        """Forward with anchor MP + skip connection.

        Returns: (first_agg, node_vec, output)
        """
        # Layer 1
        anchor_out = self.layer1(x, cur_anchor_adj, anchor_mp=True)
        if not first and update_ratio is not None and first_anchor_adj is not None:
            first_anchor_out = self.layer1(x, first_anchor_adj, anchor_mp=True)
            anchor_out = update_ratio * anchor_out + (1 - update_ratio) * first_anchor_out

        if first:
            first_agg = self.layer1(x, init_adj, anchor_mp=False)

        h = (1 - graph_skip_conn) * anchor_out + graph_skip_conn * first_agg
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        node_vec = h

        # Layer 2
        anchor_out2 = self.layer2(h, cur_anchor_adj, anchor_mp=True)
        if not first and update_ratio is not None and first_anchor_adj is not None:
            first_anchor_out2 = self.layer2(h, first_anchor_adj, anchor_mp=True)
            anchor_out2 = update_ratio * anchor_out2 + (1 - update_ratio) * first_anchor_out2

        orig_out2 = self.layer2(h, init_adj, anchor_mp=False)
        output = (1 - graph_skip_conn) * anchor_out2 + graph_skip_conn * orig_out2

        return first_agg, node_vec, output


class FairIDGL(nn.Module):
    """IDGL + FairUnGSL for fair graph structure learning.

    Normal mode: full [N, N] graph learning + FairUnGSL dense
    Scalable mode: [N, anchor_num] + FairAnchorUnGSL
    """

    def __init__(self, n_nodes, n_feat, n_class, entropy, sensitive, conf, device='cuda:0'):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_class = n_class
        self.device = device

        n_hidden = conf.get('n_hidden', 64)
        dropout = conf.get('dropout', 0.5)

        # Fair mode: "default" = 閿?閿?together (original), "D" = 閿?per-iter + 閿?final-only
        #            "G" = no 閿?閿? pure loss-driven (IGFR: per-iter graph fairness regularization)
        self.fair_mode = conf.get('fair_mode', 'default')

        # v5: hidden representation debiasing inside iteration loop
        self.debias_hidden_alpha = conf.get('debias_hidden_alpha', 0.0)  # 0 = disabled

        # v6 Mode G: graph fairness regularization weight
        self.graph_fair_weight = conf.get('graph_fair_weight', 0.0)

        # Store sensitive attribute (needed for debias_hidden and graph fair loss)
        if self.debias_hidden_alpha > 0 or self.graph_fair_weight > 0 or self.fair_mode == 'G':
            self.register_buffer('_sensitive', sensitive.float().to(device))

        # IDGL hyperparameters
        self.graph_skip_conn = conf.get('graph_skip_conn', 0.8)
        self.update_adj_ratio = conf.get('update_adj_ratio', 0.1)
        self.max_iter = conf.get('max_iter', 10)
        self.eps_adj = conf.get('eps_adj', 4e-5)
        self.feat_adj_dropout = conf.get('feat_adj_dropout', 0.5)
        self.smoothness_ratio = conf.get('smoothness_ratio', 0.2)
        self.degree_ratio = conf.get('degree_ratio', 0.0)
        self.sparsity_ratio = conf.get('sparsity_ratio', 0.0)

        # Scalable mode
        self.scalable_run = conf.get('scalable_run', n_nodes > 3000)
        self.num_anchors = conf.get('num_anchors', min(700, n_nodes // 2))

        # Graph Learners
        num_pers = conf.get('graph_learn_num_pers', 4)
        epsilon = conf.get('graph_learn_epsilon', None)
        epsilon2 = conf.get('graph_learn_epsilon2', None)
        topk = conf.get('graph_learn_topk', None)
        topk2 = conf.get('graph_learn_topk2', None)

        self.graph_learner = IDGLGraphLearner(n_feat, topk=topk, epsilon=epsilon, num_pers=num_pers)
        self.graph_learner2 = IDGLGraphLearner(n_hidden, topk=topk2, epsilon=epsilon2, num_pers=num_pers)

        # Classifier
        if self.scalable_run:
            print(f"  IDGL scalable mode (n_nodes={n_nodes}, anchors={self.num_anchors})")
            self.classifier = ScalableGCN(n_feat, n_hidden, n_class, dropout)
        else:
            print(f"  IDGL normal mode (n_nodes={n_nodes})")
            self.classifier = GCN(n_feat, n_hidden, n_class, dropout)

        # FairUnGSL module
        fair_conf = {
            'init_value': conf.get('init_value', 0.5),
            'beta': conf.get('beta', 0.1),
            'lambda_init': conf.get('lambda_init', 0.0),
            'delta_eps': conf.get('delta_eps', 0.0),
            'use_fairdrop': conf.get('use_fairdrop', False),
            'fair_mode': self.fair_mode,
            'delta_eps_init': conf.get('delta_eps_init', 0.0),
        }

        if self.scalable_run:
            self.fair_ungsl = FairAnchorUnGSL(n_nodes, entropy, sensitive, fair_conf, device)
        else:
            self.fair_ungsl = FairUnGSL(n_nodes, entropy, sensitive, fair_conf, device)

        self._adj_norm_cache = None
        self._iter_loss_cache = torch.tensor(0.0)
        self._graph_fair_loss_cache = torch.tensor(0.0)
        # Per UnGSL's README: sample anchors only in first forward, reuse in subsequent epochs.
        self._fixed_anchor_idx = None

    def _row_normalize(self, adj):
        return adj / torch.clamp(adj.sum(dim=-1, keepdim=True), min=1e-12)

    def _debias_hidden(self, h):
        """Remove sensitive attribute direction from hidden representations.

        This ensures graph_learner2 sees fairer representations,
        producing a less biased graph without multiplicative accumulation.
        """
        if self.debias_hidden_alpha <= 0:
            return h
        s = self._sensitive
        h_0 = h[s == 0].mean(dim=0)
        h_1 = h[s == 1].mean(dim=0)
        direction = h_1 - h_0
        direction = direction / direction.norm().clamp(min=1e-8)
        proj = (h @ direction).unsqueeze(1) * direction.unsqueeze(0)
        return h - self.debias_hidden_alpha * proj

    def _compute_graph_fair_loss(self, adj, anchor_idx=None):
        """Compute graph-level fairness: |avg_same_edge - avg_cross_edge|.

        Directly regularizes the adjacency matrix to have balanced connectivity
        across sensitive groups. Gradient flows back to graph_learner parameters.
        """
        s = self._sensitive

        if anchor_idx is not None:
            # Anchor mode: adj is [N, S], compare node-anchor pairs
            s_anchor = s[anchor_idx]  # [S]
            same = (s.unsqueeze(1) == s_anchor.unsqueeze(0)).float()  # [N, S]
        else:
            # Dense mode: adj is [N, N]
            same = (s.unsqueeze(0) == s.unsqueeze(1)).float()  # [N, N]

        cross = 1.0 - same
        edge_mask = (adj > 0).float()

        n_same = (same * edge_mask).sum().clamp(min=1)
        n_cross = (cross * edge_mask).sum().clamp(min=1)
        avg_same = (adj * same * edge_mask).sum() / n_same
        avg_cross = (adj * cross * edge_mask).sum() / n_cross

        return (avg_same - avg_cross).abs()

    def forward(self, x, adj):
        if self.scalable_run:
            return self._scalable_forward(x, adj)
        else:
            return self._normal_forward(x, adj)

    def _normal_forward(self, x, adj):
        """Full [N, N] mode for small graphs.

        fair_mode == "default": 閿?閿?applied at each iteration (original behavior).
        fair_mode == "D": 閿?only per-iteration, 閿?once on final output (no detach).
        fair_mode == "F": Two-phase decoupled 闁?閿?only iterations (phase 1),
                          then detach + 閿?閿?+ GCN for classification (phase 2).
        fair_mode == "G": No 閿?閿?graph modification. Fairness via per-iteration
                          graph regularization loss (IGFR). Graph learner receives
                          direct gradient signal to produce fair graphs.
        """
        if self._adj_norm_cache is None:
            with torch.no_grad():
                self._adj_norm_cache = normalize_adj(adj).detach()

        x_drop = F.dropout(x, self.feat_adj_dropout, training=self.training)

        # Mode G: no graph modification; D/F: 閿?only; default: 閿?閿?        if self.fair_mode == 'G':
            _apply = lambda a: a  # identity 闁?no 閿?閿?        elif self.fair_mode in ('D', 'F'):
            _apply = self.fair_ungsl.apply_psi_only
        else:
            _apply = self.fair_ungsl

        # Iteration 1: learn from features
        raw_adj = torch.clamp(self.graph_learner(x_drop), min=0)
        adj_fair = _apply(raw_adj)
        adj_norm = self._row_normalize(adj_fair)
        adj_fused = (1 - self.graph_skip_conn) * adj_norm + self.graph_skip_conn * self._adj_norm_cache
        adj_fused = F.dropout(adj_fused, self.feat_adj_dropout, training=self.training)

        output = self.classifier(x, adj_fused)
        node_vec = self.classifier.gc1(x, adj_fused)
        node_vec = F.relu(node_vec)

        iter_loss = compute_graph_regularization(
            raw_adj, x, self.smoothness_ratio, self.degree_ratio, self.sparsity_ratio)

        # Per-iteration graph fairness loss (works with any mode)
        graph_fair_loss = torch.tensor(0.0, device=x.device)
        if self.graph_fair_weight > 0:
            graph_fair_loss = self._compute_graph_fair_loss(raw_adj)

        first_adj_fair = adj_fair
        pre_adj = adj_fair
        n_iters = 1

        for i in range(self.max_iter - 1):
            # v5: debias hidden before feeding to graph_learner2
            node_vec_input = self._debias_hidden(node_vec.detach())
            raw_adj_new = torch.clamp(self.graph_learner2(node_vec_input), min=0)

            # Accumulate graph fair loss on raw adj (before 閿?閿?modification)
            if self.graph_fair_weight > 0:
                graph_fair_loss = graph_fair_loss + self._compute_graph_fair_loss(raw_adj_new)

            adj_fair_new = _apply(raw_adj_new)
            adj_blend = self.update_adj_ratio * adj_fair_new + (1 - self.update_adj_ratio) * first_adj_fair
            adj_norm_new = self._row_normalize(adj_blend)
            adj_fused_new = (1 - self.graph_skip_conn) * adj_norm_new + self.graph_skip_conn * self._adj_norm_cache
            adj_fused_new = F.dropout(adj_fused_new, self.feat_adj_dropout, training=self.training)

            output = self.classifier(x, adj_fused_new)
            node_vec = self.classifier.gc1(x, adj_fused_new)
            node_vec = F.relu(node_vec)

            iter_loss = iter_loss + compute_graph_regularization(
                raw_adj_new, x, self.smoothness_ratio, self.degree_ratio, self.sparsity_ratio)
            n_iters += 1

            if not self.training or diff(adj_fair_new, pre_adj, raw_adj) < self.eps_adj:
                break
            pre_adj = adj_fair_new

        self._iter_loss_cache = iter_loss / n_iters
        self._graph_fair_loss_cache = graph_fair_loss / n_iters if self.graph_fair_weight > 0 else torch.tensor(0.0, device=x.device)
        last_adj = adj_blend if self.max_iter > 1 else adj_fair

        if self.fair_mode == 'F':
            last_adj_fair = self.fair_ungsl(last_adj.detach())
            adj_norm_fair = self._row_normalize(last_adj_fair)
            adj_fused_fair = (1 - self.graph_skip_conn) * adj_norm_fair + self.graph_skip_conn * self._adj_norm_cache
            self._output_fair_cache = self.classifier(x, adj_fused_fair)
            return output, last_adj_fair, last_adj_fair
        elif self.fair_mode == 'D':
            last_adj_fair = self.fair_ungsl.apply_phi_only(last_adj)
            return output, last_adj_fair, last_adj_fair

        return output, last_adj, last_adj

    def _scalable_forward(self, x, adj):
        """Anchor-based [N, S] mode for large graphs.

        fair_mode == "D": 閿?only per-iteration, 閿?once on final output (no detach).
        fair_mode == "F": Two-phase decoupled 闁?閿?only iterations, detach + 閿?閿?+ GCN.
        fair_mode == "G": No 閿?閿? pure loss-driven (IGFR). Per-iteration graph
                          fairness regularization on [N, anchor_num] adjacency.
        """
        if self._adj_norm_cache is None:
            with torch.no_grad():
                self._adj_norm_cache = normalize_adj(adj).detach()

        x_drop = F.dropout(x, self.feat_adj_dropout, training=self.training)

        # Sample anchors (UnGSL-compatible: fix indices after first forward)
        if self._fixed_anchor_idx is None:
            _, self._fixed_anchor_idx = sample_anchors(x_drop, self.num_anchors)
        anchor_idx = self._fixed_anchor_idx
        anchor_feats = x_drop[anchor_idx]

        # Mode G: no graph modification; D/F: 閿?only; default: 閿?閿?        if self.fair_mode == 'G':
            _apply = lambda adj_m, idx: adj_m  # identity
        elif self.fair_mode in ('D', 'F'):
            _apply = lambda adj_m, idx: self.fair_ungsl.apply_psi_only(adj_m, idx)
        else:
            _apply = self.fair_ungsl

        # Iteration 1
        node_anchor_adj = torch.clamp(self.graph_learner(x_drop, anchor=anchor_feats), min=0)

        # Per-iteration graph fairness loss (works with any mode)
        graph_fair_loss = torch.tensor(0.0, device=x.device)
        if self.graph_fair_weight > 0:
            graph_fair_loss = self._compute_graph_fair_loss(node_anchor_adj, anchor_idx)

        node_anchor_adj = _apply(node_anchor_adj, anchor_idx)

        # GCN with anchor MP + skip connection
        first_agg, node_vec, output = self.classifier(
            x, self._adj_norm_cache, node_anchor_adj,
            self.graph_skip_conn, first=True)

        first_anchor_adj = node_anchor_adj
        pre_anchor_adj = node_anchor_adj
        n_iters = 1

        # Iterative refinement
        for i in range(self.max_iter - 1):
            # v5: debias hidden before feeding to graph_learner2
            node_vec_input = self._debias_hidden(node_vec.detach())
            # Use fixed anchor indices to extract hidden-layer anchor features
            anchor_vec = node_vec_input[anchor_idx]
            node_anchor_adj_new = torch.clamp(
                self.graph_learner2(node_vec_input, anchor=anchor_vec), min=0)

            # Accumulate graph fair loss on raw adj (before 閿?閿?modification)
            if self.graph_fair_weight > 0:
                graph_fair_loss = graph_fair_loss + self._compute_graph_fair_loss(
                    node_anchor_adj_new, anchor_idx)

            node_anchor_adj_new = _apply(node_anchor_adj_new, anchor_idx)

            # Momentum blend
            anchor_adj_blend = (self.update_adj_ratio * node_anchor_adj_new
                                + (1 - self.update_adj_ratio) * first_anchor_adj)

            _, node_vec, output = self.classifier(
                x, self._adj_norm_cache, anchor_adj_blend,
                self.graph_skip_conn, first=False, first_agg=first_agg,
                update_ratio=self.update_adj_ratio, first_anchor_adj=first_anchor_adj)

            n_iters += 1
            if not self.training or diff(node_anchor_adj_new, pre_anchor_adj, first_anchor_adj) < self.eps_adj:
                break
            pre_anchor_adj = node_anchor_adj_new

        self._iter_loss_cache = torch.tensor(0.0, device=x.device)
        self._graph_fair_loss_cache = graph_fair_loss / n_iters if self.graph_fair_weight > 0 else torch.tensor(0.0, device=x.device)

        last_adj = anchor_adj_blend if self.max_iter > 1 else node_anchor_adj

        if self.fair_mode == 'F':
            last_adj_fair = self.fair_ungsl(last_adj.detach(), anchor_idx)
            _, _, output_fair = self.classifier(
                x, self._adj_norm_cache, last_adj_fair,
                self.graph_skip_conn, first=False, first_agg=first_agg,
                update_ratio=self.update_adj_ratio, first_anchor_adj=first_anchor_adj)
            self._output_fair_cache = output_fair
            return output, last_adj_fair, last_adj_fair
        elif self.fair_mode == 'D':
            last_adj_fair = self.fair_ungsl.apply_phi_only(last_adj, anchor_idx)
            return output, last_adj_fair, last_adj_fair

        return output, last_adj, last_adj

    def get_iter_loss(self):
        return self._iter_loss_cache

    def get_graph_fair_loss(self):
        """Return accumulated per-iteration graph fairness loss (Mode G only)."""
        return self._graph_fair_loss_cache

    def base_parameters(self):
        return list(self.classifier.parameters())

    def graph_parameters(self):
        return list(self.graph_learner.parameters()) + list(self.graph_learner2.parameters())

    def ungsl_parameters(self):
        return self.fair_ungsl.ungsl_parameters()

    def fairness_parameters(self):
        return self.fair_ungsl.fairness_parameters()
