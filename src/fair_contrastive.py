"""
Fair Contrastive Learning module for FairUnGSL.

Based on:
- FairSCR (arXiv 2404.06090, 2024): Supervised Contrastive Loss on disentangled content
- Khosla et al. (NeurIPS 2020): Supervised Contrastive Learning (SupCon)
- FCLCA (KBS 2024): Counterfactual graph contrastive

Design choice: Sensitive-aware SupCon
  - Positive pair (i, j): same label AND different sensitive
    → Pull minority and majority group nodes of the same label closer
  - Negative: all other nodes
    → Push away same-label-same-group (avoid group homogeneity)
    → Push away different-label

This is MORE aggressive than FairSCR's sensitive-agnostic SupCon because
it directly injects the fairness objective into positive-pair definition.

Integration: operates on backbone's hidden representation h via 2-layer MLP
projection head (as recommended by SimCLR ablation studies).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairContrastiveHead(nn.Module):
    """
    Projection head + Fair Supervised Contrastive Loss.

    Backbone-agnostic: plugs into any GSL backbone via the output
    hidden representation h [N, hidden_dim].

    Parameters
    ----------
    hidden_dim : int
        Input dimension (backbone's node embedding dim).
    proj_dim : int
        Projection output dim (default 64).
    temperature : float
        InfoNCE temperature (default 0.5).
    mode : str
        'supcon_sens' — same-label-diff-sens positives (recommended)
        'supcon_label' — same-label positives (FairSCR style baseline)
        'counterfactual' — original vs counterfactual-sensitive view
    """

    def __init__(self, hidden_dim, proj_dim=64, temperature=0.5, mode='supcon_sens'):
        super().__init__()
        self.mode = mode
        self.temperature = temperature

        # 2-layer MLP projection (SimCLR / Graphair style)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, h, labels, sensitive, mask):
        """
        Compute fair contrastive loss.

        Parameters
        ----------
        h : torch.Tensor [N, hidden_dim]
            Backbone hidden representations.
        labels : torch.Tensor [N] (long)
            Ground truth class labels.
        sensitive : torch.Tensor [N] (long, 0/1)
            Binary sensitive attribute.
        mask : torch.Tensor [N] (bool)
            Training mask — only compute loss on training nodes.

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        # Restrict to training nodes
        h_train = h[mask]
        y_train = labels[mask]
        s_train = sensitive[mask]
        n = h_train.size(0)

        if n < 2:
            return torch.tensor(0.0, device=h.device)

        # Projection + L2 normalize
        z = self.projection(h_train)
        z = F.normalize(z, dim=-1)  # [n, proj_dim]

        # Similarity matrix
        sim = (z @ z.T) / self.temperature  # [n, n]

        # Numerical stability: subtract row max (standard SupCon trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask out self-similarity (diagonal)
        logits_mask = 1.0 - torch.eye(n, device=h.device)

        # Build positive mask based on mode
        if self.mode == 'supcon_sens':
            # Positive: same label AND different sensitive
            same_label = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()
            diff_sens = (s_train.unsqueeze(0) != s_train.unsqueeze(1)).float()
            pos_mask = same_label * diff_sens * logits_mask
        elif self.mode == 'supcon_label':
            # FairSCR-style: positive = same label
            same_label = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()
            pos_mask = same_label * logits_mask
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Skip if no positives (e.g., all nodes same sensitive)
        pos_count = pos_mask.sum(dim=1)  # [n]
        valid = pos_count > 0  # nodes that have at least one positive
        if valid.sum() == 0:
            return torch.tensor(0.0, device=h.device)

        # SupCon loss (Khosla 2020, Eq. 2):
        #   L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(sim_ip) / sum_{a != i} exp(sim_ia) )
        exp_sim = torch.exp(sim) * logits_mask  # [n, n], zero diagonal

        log_prob_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)  # [n, 1]
        log_prob = sim - log_prob_denom  # log p(a|i) for all a

        # Mean log-prob over positives, only for valid nodes
        pos_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        loss = -pos_log_prob[valid].mean()

        return loss
