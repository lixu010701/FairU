"""
FairPROSE: PROSE + FairUnGSL integration.

PROSE learns graph structure via anchor-based multi-stage graph learning.
Unlike PROGNN (O(N閾? parameter) or GRCN (feature-based similarity),
PROSE uses anchor nodes and cosine similarity to learn a [N, anchor_num]
node-anchor adjacency matrix 闁?memory efficient for all graph sizes.

FairUnGSL hooks into the node-anchor adjacency via FairAnchorUnGSL,
applying the same v4 fairness modulation (threshold + 閿?+ Direction B + FairDrop)
adapted to the [N, anchor_num] format.
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_BACKBONE_DIR = os.path.dirname(__file__)
if _BACKBONE_DIR not in sys.path:
    sys.path.insert(0, _BACKBONE_DIR)

from graph_learners import Stage_GNN_learner
from model import GCN_Sparse, Anchor_GCL
from utils import (get_feat_mask, split_batch, normalize, torch_sparse_eye,
                   compute_anchor_adj, add_graph_degree_loss)


class FairAnchorUnGSL(nn.Module):
    """FairUnGSL adapted for anchor-based adjacency [N, anchor_num].

    Applies v4 fairness-aware edge refinement to node-anchor adjacency.
    Each call receives anchor_nodes_idx to identify which nodes are anchors.
    """

    def __init__(self, n_nodes, entropy, sensitive, conf, device='cuda:0'):
        super().__init__()

        self.fair_mode = conf.get('fair_mode', 'default')

        self.thresholds = nn.Parameter(torch.FloatTensor(n_nodes, 1))
        self.thresholds.data.fill_(conf.get('init_value', 0.5))
        self.Beta = conf.get('beta', 0.1)

        confidence = torch.exp(-entropy).to(device)
        self.register_buffer('confidence_vector', confidence)
        self.register_buffer('uncertainty_vector', 1.0 - confidence)
        self.register_buffer('sensitive', sensitive.float().to(device))

        if self.fair_mode == 'E':
            self.delta_eps_learnable = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.delta_eps_learnable.data.fill_(conf.get('delta_eps_init', 0.0))
            self.delta_eps = 0.0
        else:
            self.lambdas = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.lambdas.data.fill_(conf.get('lambda_init', 0.0))
            self.delta_eps = float(conf.get('delta_eps', 0.0))

        self.use_fairdrop = bool(conf.get('use_fairdrop', False))

    def forward(self, node_anchor_adj, anchor_nodes_idx):
        """Apply FairUnGSL to node-anchor adjacency.

        Parameters
        ----------
        node_anchor_adj : torch.Tensor [N, anchor_num]
            Node-anchor adjacency matrix.
        anchor_nodes_idx : torch.Tensor [anchor_num]
            Indices of anchor nodes.

        Returns
        -------
        torch.Tensor [N, anchor_num]
            Refined node-anchor adjacency.
        """
        n_nodes = node_anchor_adj.shape[0]

        # Confidence/uncertainty of anchor nodes: [anchor_num] 闁?broadcast to [N, anchor_num]
        conf_anchors = self.confidence_vector[anchor_nodes_idx].unsqueeze(0)  # [1, anchor_num]
        uncert_anchors = self.uncertainty_vector[anchor_nodes_idx].unsqueeze(0)  # [1, anchor_num]

        # Cross-group indicator: [N, anchor_num]
        sens_nodes = self.sensitive.unsqueeze(1)  # [N, 1]
        sens_anchors = self.sensitive[anchor_nodes_idx].unsqueeze(0)  # [1, anchor_num]
        delta = (sens_nodes - sens_anchors).abs()  # [N, anchor_num]

        # Threshold computation
        if self.fair_mode == 'E':
            eps_per_edge = self.thresholds - self.delta_eps_learnable * delta
        else:
            eps_per_edge = self.thresholds - self.delta_eps * delta

        # Confidence-based filtering
        weight = torch.sigmoid(conf_anchors - eps_per_edge) / 0.5

        # v4 Direction A: FairDrop for same-group failed edges (training only)
        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (delta < 0.5)
            beta_full = torch.full_like(weight, self.Beta)
            mask = torch.where(
                failed & same_group, bernoulli_kept,
                torch.where(failed, beta_full, weight),
            )
        else:
            mask = torch.where(weight >= 1, weight, self.Beta)

        if self.fair_mode == 'E':
            return node_anchor_adj * mask
        else:
            lambda_vals = torch.sigmoid(self.lambdas)
            phi = 1.0 - lambda_vals * delta * uncert_anchors
            return node_anchor_adj * mask * phi

    def apply_psi_only(self, node_anchor_adj, anchor_nodes_idx):
        """Apply only 閿?(confidence-based filtering), without 閿?

        Mode D: used during iterative graph learning.
        """
        conf_anchors = self.confidence_vector[anchor_nodes_idx].unsqueeze(0)
        sens_nodes = self.sensitive.unsqueeze(1)
        sens_anchors = self.sensitive[anchor_nodes_idx].unsqueeze(0)
        delta = (sens_nodes - sens_anchors).abs()

        eps_per_edge = self.thresholds - self.delta_eps * delta
        weight = torch.sigmoid(conf_anchors - eps_per_edge) / 0.5

        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (delta < 0.5)
            beta_full = torch.full_like(weight, self.Beta)
            mask = torch.where(
                failed & same_group, bernoulli_kept,
                torch.where(failed, beta_full, weight),
            )
        else:
            mask = torch.where(weight >= 1, weight, self.Beta)

        return node_anchor_adj * mask

    def apply_phi_only(self, node_anchor_adj, anchor_nodes_idx):
        """Apply only 閿?(fairness modulation), without 閿?

        Mode D: used once on the final graph output.
        """
        sens_nodes = self.sensitive.unsqueeze(1)
        sens_anchors = self.sensitive[anchor_nodes_idx].unsqueeze(0)
        delta = (sens_nodes - sens_anchors).abs()

        uncert_anchors = self.uncertainty_vector[anchor_nodes_idx].unsqueeze(0)

        lambda_vals = torch.sigmoid(self.lambdas)
        phi = 1.0 - lambda_vals * delta * uncert_anchors

        return node_anchor_adj * phi

    def ungsl_parameters(self):
        return [self.thresholds]

    def fairness_parameters(self):
        if self.fair_mode == 'E':
            return [self.delta_eps_learnable]
        return [self.lambdas]


class FairPROSE(nn.Module):
    """PROSE + FairUnGSL model for fair graph structure learning.

    Architecture:
    1. Stage_GNN_learner: anchor-based multi-stage graph learning
    2. FairAnchorUnGSL: fairness-aware edge refinement on [N, anchor_num]
    3. GCN_Sparse classifier: node classification with anchor message passing
    4. Anchor_GCL: contrastive learning (optional)
    """

    def __init__(self, n_nodes, n_feat, n_class, entropy, sensitive, conf, device='cuda:0'):
        super().__init__()

        self.n_nodes = n_nodes
        self.device = device

        # --- PROSE hyperparameters ---
        self.anchor_num = conf.get('anchor_num', min(700, n_nodes // 2))
        self.emb_fusion_ratio = conf.get('emb_fusion_ratio', 0.35)
        self.anchor_weight = conf.get('anchor_weight', 0.03)
        self.tau = conf.get('tau', 1.0)  # structure bootstrapping weight (1.0 = no bootstrap)
        self.bootstrap_interval = conf.get('bootstrap_interval', 0)
        self.head_tail_mi = conf.get('head_tail_mi', 1)
        self.mi_ratio = conf.get('mi_ratio', 0.1)
        self.maskfeat_rate_anchor = conf.get('maskfeat_rate_anchor', 0.2)
        self.maskfeat_rate_learner = conf.get('maskfeat_rate_learner', 0.2)
        self.contrast_batch_size = conf.get('contrast_batch_size', 0)

        n_hidden = conf.get('n_hidden', 64)
        dropout = conf.get('dropout', 0.5)

        # --- Graph Learner ---
        stage_ks = conf.get('stage_ks', [])
        self.graph_learner = Stage_GNN_learner(
            isize=n_feat,
            osize=conf.get('hidden_dim_cls', n_hidden),
            head_num=conf.get('head_num', 6),
            sparse=True,
            ks=stage_ks,
            anchor_adj_fusion_ratio=conf.get('anchor_adj_fusion_ratio', 0.95),
            epsilon=conf.get('epsilon', 0.1),
        )

        # --- Classifier ---
        self.classifier = GCN_Sparse(
            nfeat=n_feat,
            nhid=conf.get('hidden_dim_cls', n_hidden),
            nclass=n_class,
            graph_hops=conf.get('nlayers_cls', 2),
            dropout=conf.get('dropout_cls', dropout),
            batch_norm=conf.get('bn_cls', False),
        )

        # --- Contrastive Learning ---
        if self.head_tail_mi:
            self.contrastive = Anchor_GCL(
                nlayers=conf.get('nlayers', 2),
                in_dim=n_feat,
                hidden_dim=conf.get('hidden_dim', n_hidden),
                emb_dim=conf.get('rep_dim', 32),
                proj_dim=conf.get('proj_dim', 16),
                dropout=dropout,
                dropout_adj=conf.get('dropedge_rate', 0.25),
            )
        else:
            self.contrastive = None

        # Fair mode
        self.fair_mode = conf.get('fair_mode', 'default')

        # --- FairAnchorUnGSL module ---
        fair_conf = {
            'init_value': conf.get('init_value', 0.5),
            'beta': conf.get('beta', 0.1),
            'lambda_init': conf.get('lambda_init', 0.0),
            'delta_eps': conf.get('delta_eps', 0.0),
            'use_fairdrop': conf.get('use_fairdrop', False),
            'fair_mode': self.fair_mode,
            'delta_eps_init': conf.get('delta_eps_init', 0.0),
        }

        print(f"  Using FairAnchorUnGSL (anchor_num={self.anchor_num}, fair_mode={self.fair_mode})")
        self.fair_ungsl = FairAnchorUnGSL(n_nodes, entropy, sensitive, fair_conf, device)

        # Cached normalized anchor adjacency
        self._anchor_adj = None
        self._adj_norm_cache = None  # interface compat with FairGRCN (contrastive path in train.py)
        self._epoch = 0

    def _ensure_anchor_adj(self, adj):
        """Initialize and cache the normalized anchor adjacency."""
        if self._anchor_adj is None:
            with torch.no_grad():
                if adj.is_sparse:
                    adj_with_loop = adj + torch_sparse_eye(adj.shape[0]).to(adj.device)
                else:
                    adj_with_loop = adj + torch.eye(adj.shape[0], device=adj.device)
                self._anchor_adj = normalize(adj_with_loop, 'sym', adj.is_sparse)

    def forward(self, x, adj):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor [N, F]
            Node features.
        adj : torch.Tensor [N, N]
            Original adjacency (sparse).

        Returns
        -------
        output : torch.Tensor [N, C]
            Classification logits.
        node_anchor_adj : torch.Tensor [N, anchor_num]
            Learned node-anchor adjacency after fairness refinement.
        node_anchor_adj : torch.Tensor
            Same as above (for interface compatibility).
        """
        self._ensure_anchor_adj(adj)
        self._epoch += 1

        # 1. Select random anchor nodes
        anchor_nodes_idx = torch.randperm(x.size(0), device=x.device)[:self.anchor_num]

        # 2. Graph learning: produce node-anchor adjacency [N, anchor_num]
        node_anchor_adj = self.graph_learner.forward_anchor(
            x, self._anchor_adj, anchor_nodes_idx,
            self.classifier.graph_encoders[0],
            self.emb_fusion_ratio,
        )

        # 3. FairUnGSL refinement on [N, anchor_num]
        if self.fair_mode == 'F':
            # Mode F: 閿?only on learned adj, then detach + 閿?閿?for classification
            node_anchor_adj = self.fair_ungsl.apply_psi_only(node_anchor_adj, anchor_nodes_idx)
            node_anchor_adj_fair = self.fair_ungsl(node_anchor_adj.detach(), anchor_nodes_idx)
        elif self.fair_mode == 'D':
            node_anchor_adj = self.fair_ungsl.apply_psi_only(node_anchor_adj, anchor_nodes_idx)
            node_anchor_adj_fair = self.fair_ungsl.apply_phi_only(node_anchor_adj, anchor_nodes_idx)
        else:
            node_anchor_adj = self.fair_ungsl(node_anchor_adj, anchor_nodes_idx)
            node_anchor_adj_fair = node_anchor_adj

        # Cache for contrastive loss and bootstrapping
        self._last_anchor_nodes_idx = anchor_nodes_idx
        self._last_node_anchor_adj = node_anchor_adj_fair

        # 4. Classification with anchor message passing + fusion (using 閿?adj)
        output = self._classify_with_fusion(x, node_anchor_adj_fair)

        # 5. For small graphs: convert [N, anchor_num] 闁?approximate [N, N] for fairness loss
        # For large graphs (>5000): skip to avoid O(N閾? memory
        if self.n_nodes <= 5000:
            adj_approx = compute_anchor_adj(node_anchor_adj_fair)
            self._degree_loss = add_graph_degree_loss(adj_approx, self.anchor_weight)
            return output, adj_approx, adj_approx
        else:
            self._degree_loss = torch.tensor(0.0, device=x.device)
            return output, node_anchor_adj_fair, node_anchor_adj_fair

    def _classify_with_fusion(self, x, node_anchor_adj):
        """Classification using anchor message passing fused with original adj."""
        fusion = self.emb_fusion_ratio
        if fusion == 0.0:
            return self.classifier(x, self._anchor_adj)

        encoders = self.classifier.graph_encoders

        # First layer
        anchor_out = encoders[0](x, node_anchor_adj, anchor_mp=True, batch_norm=False)
        orig_out = encoders[0](x, self._anchor_adj, anchor_mp=False, batch_norm=False)
        h = fusion * anchor_out + (1 - fusion) * orig_out
        if encoders[0].bn is not None:
            h = encoders[0].bn(h)
        h = F.relu(h)
        h = F.dropout(h, self.classifier.dropout, training=self.training)

        # Middle layers
        for encoder in encoders[1:-1]:
            anchor_out = encoder(h, node_anchor_adj, anchor_mp=True, batch_norm=False)
            orig_out = encoder(h, self._anchor_adj, anchor_mp=False, batch_norm=False)
            h = fusion * anchor_out + (1 - fusion) * orig_out
            if encoder.bn is not None:
                h = encoder.bn(h)
            h = F.relu(h)
            h = F.dropout(h, self.classifier.dropout, training=self.training)

        # Last layer
        anchor_out = encoders[-1](h, node_anchor_adj, anchor_mp=True, batch_norm=False)
        orig_out = encoders[-1](h, self._anchor_adj, anchor_mp=False, batch_norm=False)
        output = fusion * anchor_out + (1 - fusion) * orig_out

        return output

    def compute_contrastive_loss(self, x):
        """Compute contrastive loss between anchor view and learned view.

        Must be called AFTER forward().
        """
        if self.contrastive is None or not self.head_tail_mi:
            return torch.tensor(0.0, device=x.device)

        # View 1: anchor graph (original adj with feature masking)
        if self.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(x, self.maskfeat_rate_anchor)
            x_v1 = x * (1 - mask_v1)
        else:
            x_v1 = x

        z1, _ = self.contrastive(x_v1, self._anchor_adj, anchor_mp=False, batch_norm=False)

        # View 2: learned graph (with feature masking)
        if self.maskfeat_rate_learner:
            mask_v2, _ = get_feat_mask(x, self.maskfeat_rate_learner)
            x_v2 = x * (1 - mask_v2)
        else:
            x_v2 = x

        z2, _ = self.contrastive(
            x_v2, self._last_node_anchor_adj, anchor_mp=True, batch_norm=False
        )

        # Compute contrastive loss (with batching if configured)
        if self.contrast_batch_size:
            node_idxs = list(range(x.shape[0]))
            batches = split_batch(node_idxs, self.contrast_batch_size)
            loss = torch.tensor(0.0, device=x.device)
            for batch in batches:
                weight = len(batch) / x.shape[0]
                loss = loss + Anchor_GCL.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = Anchor_GCL.calc_loss(z1, z2)

        return self.mi_ratio * loss

    def maybe_bootstrap(self, epoch):
        """Structure bootstrapping: update anchor_adj with learned structure."""
        if self.tau < 1.0 and self._anchor_adj is not None:
            if self.bootstrap_interval == 0 or epoch % self.bootstrap_interval == 0:
                with torch.no_grad():
                    learned_full = compute_anchor_adj(self._last_node_anchor_adj)
                    self._anchor_adj = (
                        self._anchor_adj.to_dense() * self.tau
                        + learned_full.detach() * (1 - self.tau)
                    )

    def get_degree_loss(self):
        """Return degree regularization loss from last forward."""
        return self._degree_loss if hasattr(self, '_degree_loss') else torch.tensor(0.0)

    def base_parameters(self):
        """Classifier parameters."""
        return list(self.classifier.parameters())

    def graph_parameters(self):
        """Graph learner parameters."""
        return list(self.graph_learner.parameters())

    def ungsl_parameters(self):
        """UnGSL threshold parameters."""
        return self.fair_ungsl.ungsl_parameters()

    def fairness_parameters(self):
        """Fairness strength parameters."""
        return self.fair_ungsl.fairness_parameters()

    def contrastive_parameters(self):
        """Contrastive learning parameters."""
        if self.contrastive is not None:
            return list(self.contrastive.parameters())
        return []
