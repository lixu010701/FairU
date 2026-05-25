"""
FairUnGSL: Fairness-Aware Uncertainty-aware Graph Structure Learning

Core module implementing the fairness-aware edge refinement mechanism.
Extends UnGSL's entropy-threshold framework with a fairness modulation factor.

Original UnGSL formula (Eq. 10):
    Ŝ_ij = S_ij · ψ(e^{-u_j} − ε_i)

FairUnGSL extension (default mode, for GRCN):
    Ŝ_ij = S_ij · ψ(e^{-u_j} − ε_i) · φ_ij
    φ_ij = 1 − σ(λ_i) · Δ_ij · (1 − e^{-u_j})

Mode E (for iterative backbones: IDGL/PROGNN/PROSE):
    Ŝ_ij = S_ij · ψ(e^{-u_j} − ε_ij)
    ε_ij = ε_i − δε_i · Δ_ij   (δε_i is learnable per-node)
    No φ — fairness is embedded in ψ via learnable δε.

where:
    - Δ_ij = |s_i − s_j| (sensitive attribute distance)
    - λ_i = learnable per-node fairness strength (default mode)
    - δε_i = learnable per-node cross-group threshold offset (mode E)
    - (1 − e^{-u_j}) = uncertainty of node j
"""

import torch
import torch.nn as nn


class FairSparseUnGSL(nn.Module):
    """
    Sparse version of FairUnGSL for efficient edge-level operations.
    Suitable for methods like SLAPS, IDGL, PROSE that produce sparse adjacency.
    """

    def __init__(self, n_nodes, entropy, sensitive, conf, device='cuda:0'):
        """
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the graph.
        entropy : torch.Tensor
            Pre-computed node entropy vector, shape [n_nodes].
        sensitive : torch.Tensor
            Binary sensitive attribute vector, shape [n_nodes].
        conf : dict
            Configuration dict with keys: init_value, beta, lambda_init.
        device : str
            Device to use.
        """
        super(FairSparseUnGSL, self).__init__()

        self.fair_mode = conf.get('fair_mode', 'default')

        # --- Original UnGSL components ---
        self.thresholds = nn.Parameter(torch.FloatTensor(n_nodes, 1))
        self.thresholds.data.fill_(conf['init_value'])
        self.Beta = conf['beta']

        # Confidence and uncertainty from entropy
        confidence = torch.exp(-entropy).to(device)
        self.register_buffer('confidence_vector', confidence)
        self.register_buffer('uncertainty_vector', 1.0 - confidence)

        # --- Sensitive attribute ---
        self.register_buffer('sensitive', sensitive.float().to(device))

        # --- Mode-dependent fairness mechanism ---
        if self.fair_mode == 'E':
            # Mode E: learnable per-node δε replaces φ
            # δε_i controls how much the threshold is lowered for cross-group edges of node i
            self.delta_eps_learnable = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.delta_eps_learnable.data.fill_(conf.get('delta_eps_init', 0.0))
            self.delta_eps = 0.0  # disable fixed delta_eps
        else:
            # Default mode: fixed δε + φ (for GRCN)
            self.lambdas = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.lambdas.data.fill_(conf.get('lambda_init', 0.0))
            self.delta_eps = float(conf.get('delta_eps', 0.0))

        # v4 Direction A: FairDrop-style hard random drop for SAME-GROUP failed edges
        self.use_fairdrop = bool(conf.get('use_fairdrop', False))
        # Ablation: disable phi when use_phi=False
        self.use_phi = bool(conf.get('use_phi', True))

    def forward(self, learned_adj):
        """
        Apply FairUnGSL refinement to a learned adjacency matrix.

        Parameters
        ----------
        learned_adj : torch.Tensor or tuple
            Sparse/dense adjacency, OR edge-level tuple (indices, values, shape)
            from SparseGraphDiffusion.

        Returns
        -------
        tuple (indices, new_values, shape) or sparse tensor
            Same format as input — tuple if input was tuple, sparse otherwise.
        """
        # Support edge-level tuple input (from SparseGraphDiffusion)
        if isinstance(learned_adj, tuple):
            indices, values, shape = learned_adj
            src, dst = indices[0], indices[1]
            return_tuple = True
        else:
            if not learned_adj.is_sparse:
                learned_adj = learned_adj.to_sparse().coalesce()
            else:
                learned_adj = learned_adj.coalesce()
            indices = learned_adj.indices()
            values = learned_adj.values()
            shape = learned_adj.shape
            src, dst = indices[0], indices[1]
            return_tuple = False

        # --- per-edge data ---
        confidence_dst = self.confidence_vector[dst]
        delta = (self.sensitive[src] - self.sensitive[dst]).abs()  # 1 if cross-group

        # --- Compute per-edge threshold ---
        eps_src = self.thresholds[src].flatten()
        if self.fair_mode == 'E':
            # Mode E: learnable per-node δε for cross-group edges
            deps = self.delta_eps_learnable[src].flatten()
            eps_per_edge = eps_src - deps * delta
        else:
            # Default: fixed scalar δε
            eps_per_edge = eps_src - self.delta_eps * delta

        # --- UnGSL: confidence-based filtering ---
        weight = torch.sigmoid(confidence_dst - eps_per_edge) / 0.5

        # v4 Direction A: FairDrop-style hard random drop for same-group failed edges (training only)
        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (delta < 0.5)
            beta_per_edge = torch.full_like(weight, self.Beta)
            masks = torch.where(
                failed & same_group,
                bernoulli_kept,
                torch.where(failed, beta_per_edge, weight),
            )
        else:
            masks = torch.where(weight >= 1, weight, self.Beta)

        # --- Apply fairness mechanism ---
        if self.fair_mode == 'E' or not self.use_phi:
            # Mode E: no φ, fairness is embedded in ψ via learnable δε
            new_values = values * masks
        else:
            # Default: apply φ modulation
            uncertainty_dst = self.uncertainty_vector[dst]
            lambda_src = torch.sigmoid(self.lambdas[src].flatten())
            phi = 1.0 - lambda_src * delta * uncertainty_dst
            new_values = values * masks * phi

        if return_tuple:
            return (indices, new_values, shape)
        else:
            result = torch.sparse_coo_tensor(indices, new_values, shape)
            return result

    def apply_psi_only(self, learned_adj):
        """Apply only ψ (confidence-based filtering), without φ.

        Mode D: used during iterative graph learning (same as baseline UnGSL).
        """
        if isinstance(learned_adj, tuple):
            indices, values, shape = learned_adj
            src, dst = indices[0], indices[1]
            return_tuple = True
        else:
            if not learned_adj.is_sparse:
                learned_adj = learned_adj.to_sparse().coalesce()
            else:
                learned_adj = learned_adj.coalesce()
            indices = learned_adj.indices()
            values = learned_adj.values()
            shape = learned_adj.shape
            src, dst = indices[0], indices[1]
            return_tuple = False

        confidence_dst = self.confidence_vector[dst]
        delta = (self.sensitive[src] - self.sensitive[dst]).abs()

        eps_src = self.thresholds[src].flatten()
        eps_per_edge = eps_src - self.delta_eps * delta

        weight = torch.sigmoid(confidence_dst - eps_per_edge) / 0.5

        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (delta < 0.5)
            beta_per_edge = torch.full_like(weight, self.Beta)
            masks = torch.where(
                failed & same_group, bernoulli_kept,
                torch.where(failed, beta_per_edge, weight),
            )
        else:
            masks = torch.where(weight >= 1, weight, self.Beta)

        new_values = values * masks

        if return_tuple:
            return (indices, new_values, shape)
        else:
            return torch.sparse_coo_tensor(indices, new_values, shape)

    def apply_phi_only(self, learned_adj):
        """Apply only φ (fairness modulation), without ψ.

        Mode D: used once on the final graph output.
        """
        if isinstance(learned_adj, tuple):
            indices, values, shape = learned_adj
            src, dst = indices[0], indices[1]
            return_tuple = True
        else:
            if not learned_adj.is_sparse:
                learned_adj = learned_adj.to_sparse().coalesce()
            else:
                learned_adj = learned_adj.coalesce()
            indices = learned_adj.indices()
            values = learned_adj.values()
            shape = learned_adj.shape
            src, dst = indices[0], indices[1]
            return_tuple = False

        delta = (self.sensitive[src] - self.sensitive[dst]).abs()
        uncertainty_dst = self.uncertainty_vector[dst]

        lambda_src = torch.sigmoid(self.lambdas[src].flatten())
        phi = 1.0 - lambda_src * delta * uncertainty_dst

        new_values = values * phi

        if return_tuple:
            return (indices, new_values, shape)
        else:
            return torch.sparse_coo_tensor(indices, new_values, shape)

    def ungsl_parameters(self):
        """Return UnGSL threshold parameters."""
        return [self.thresholds]

    def fairness_parameters(self):
        """Return fairness parameters (mode-dependent)."""
        if self.fair_mode == 'E':
            return [self.delta_eps_learnable]
        return [self.lambdas]


class FairUnGSL(nn.Module):
    """
    Dense version of FairUnGSL using full confidence matrix.
    Suitable for methods like GRCN that use dense adjacency operations.
    """

    def __init__(self, n_nodes, entropy, sensitive, conf, device='cuda:0'):
        """
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the graph.
        entropy : torch.Tensor
            Pre-computed node entropy vector, shape [n_nodes].
        sensitive : torch.Tensor
            Binary sensitive attribute vector, shape [n_nodes].
        conf : dict
            Configuration dict with keys: init_value, beta, lambda_init.
        device : str
            Device to use.
        """
        super(FairUnGSL, self).__init__()

        self.fair_mode = conf.get('fair_mode', 'default')

        # --- Original UnGSL components ---
        self.thresholds = nn.Parameter(torch.FloatTensor(n_nodes, 1))
        self.thresholds.data.fill_(conf['init_value'])
        self.Beta = conf['beta']

        # Build confidence matrix [N, N] — position [i,j] has confidence of node j
        confidence = torch.exp(-entropy)
        confidence_matrix = confidence.view(-1, 1).expand(-1, n_nodes).t().contiguous().to(device)
        uncertainty_matrix = (1.0 - confidence).view(-1, 1).expand(-1, n_nodes).t().contiguous().to(device)
        self.register_buffer('confidence_matrix', confidence_matrix)
        self.register_buffer('uncertainty_matrix', uncertainty_matrix)

        # Delta matrix [N, N] — 1 if cross-group, 0 if same-group
        s = sensitive.float().to(device)
        delta_matrix = (s.unsqueeze(0) - s.unsqueeze(1)).abs()
        self.register_buffer('delta_matrix', delta_matrix)

        # --- Mode-dependent fairness mechanism ---
        if self.fair_mode == 'E':
            # Mode E: learnable per-node δε replaces φ
            self.delta_eps_learnable = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.delta_eps_learnable.data.fill_(conf.get('delta_eps_init', 0.0))
            self.delta_eps = 0.0
        else:
            # Default mode: fixed δε + φ (for GRCN)
            self.lambdas = nn.Parameter(torch.FloatTensor(n_nodes, 1))
            self.lambdas.data.fill_(conf.get('lambda_init', 0.0))
            self.delta_eps = float(conf.get('delta_eps', 0.0))

        # v4 Direction A: FairDrop-style hard random drop flag
        self.use_fairdrop = bool(conf.get('use_fairdrop', False))
        # Ablation: disable phi when use_phi=False
        self.use_phi = bool(conf.get('use_phi', True))

    def forward(self, learned_adj):
        """
        Apply FairUnGSL refinement to a learned adjacency matrix.

        Parameters
        ----------
        learned_adj : torch.Tensor
            Learned adjacency matrix (dense or sparse).

        Returns
        -------
        torch.Tensor
            Refined adjacency matrix (dense).
        """
        if learned_adj.is_sparse:
            learned_adj = learned_adj.to_dense()

        edge_mask = (learned_adj > 0).float()

        # --- Compute per-edge threshold ---
        if self.fair_mode == 'E':
            eps_per_edge = self.thresholds - self.delta_eps_learnable * self.delta_matrix
        else:
            eps_per_edge = self.thresholds - self.delta_eps * self.delta_matrix

        # --- UnGSL: confidence-based filtering ---
        confidence = self.confidence_matrix * edge_mask
        weight = torch.sigmoid(confidence - eps_per_edge) / 0.5

        # v4 Direction A: FairDrop-style hard random drop for same-group failed edges (training only)
        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (self.delta_matrix < 0.5)
            beta_full = torch.full_like(weight, self.Beta)
            mask = torch.where(
                failed & same_group,
                bernoulli_kept,
                torch.where(failed, beta_full, weight),
            )
        else:
            mask = torch.where(weight >= 1, weight, self.Beta)

        # --- Apply fairness mechanism ---
        if self.fair_mode == 'E' or not self.use_phi:
            learned_adj = learned_adj * mask
        else:
            uncertainty = self.uncertainty_matrix * edge_mask
            lambda_vals = torch.sigmoid(self.lambdas)
            phi = 1.0 - lambda_vals * self.delta_matrix * uncertainty
            learned_adj = learned_adj * mask * phi

        return learned_adj

    def apply_psi_only(self, learned_adj):
        """Apply only ψ (confidence-based filtering), without φ.

        Mode D: used during iterative graph learning (same as baseline UnGSL).
        """
        if learned_adj.is_sparse:
            learned_adj = learned_adj.to_dense()

        edge_mask = (learned_adj > 0).float()

        eps_per_edge = self.thresholds - self.delta_eps * self.delta_matrix
        confidence = self.confidence_matrix * edge_mask
        weight = torch.sigmoid(confidence - eps_per_edge) / 0.5

        if self.use_fairdrop and self.training:
            bernoulli_kept = torch.bernoulli(torch.full_like(weight, self.Beta))
            failed = (weight < 1)
            same_group = (self.delta_matrix < 0.5)
            beta_full = torch.full_like(weight, self.Beta)
            mask = torch.where(
                failed & same_group, bernoulli_kept,
                torch.where(failed, beta_full, weight),
            )
        else:
            mask = torch.where(weight >= 1, weight, self.Beta)

        return learned_adj * mask

    def apply_phi_only(self, learned_adj):
        """Apply only φ (fairness modulation), without ψ.

        Mode D: used once on the final graph output.
        """
        if learned_adj.is_sparse:
            learned_adj = learned_adj.to_dense()

        edge_mask = (learned_adj > 0).float()
        uncertainty = self.uncertainty_matrix * edge_mask

        lambda_vals = torch.sigmoid(self.lambdas)
        phi = 1.0 - lambda_vals * self.delta_matrix * uncertainty

        return learned_adj * phi

    def ungsl_parameters(self):
        """Return UnGSL threshold parameters."""
        return [self.thresholds]

    def fairness_parameters(self):
        """Return fairness parameters (mode-dependent)."""
        if self.fair_mode == 'E':
            return [self.delta_eps_learnable]
        return [self.lambdas]
