"""
Fairness loss functions for FairUnGSL.

Includes:
- Statistical Parity regularization
- Structure-level fairness regularization
- Curriculum scheduling for fairness-utility balance
"""

import torch
import torch.nn.functional as F


class FairnessLoss(torch.nn.Module):
    """
    Combined fairness loss with curriculum scheduling.

    L_fair = alpha_sp * L_SP + alpha_struct * L_struct
    L_total = L_task + alpha(t) * L_fair

    where alpha(t) = alpha_max * min(1.0, epoch / warmup_epochs)
    """

    def __init__(self, alpha_sp=1.0, alpha_struct=0.5, alpha_max=1.0, warmup_epochs=50,
                 alpha_eo=0.0):
        """
        Parameters
        ----------
        alpha_sp : float
            Weight for statistical parity loss.
        alpha_struct : float
            Weight for structure fairness loss.
        alpha_max : float
            Maximum fairness loss weight (reached after warmup).
        warmup_epochs : int
            Number of epochs to linearly ramp up fairness weight.
        alpha_eo : float
            Weight for equal opportunity loss (0 = disabled for backward compat).
        """
        super(FairnessLoss, self).__init__()
        self.alpha_sp = alpha_sp
        self.alpha_struct = alpha_struct
        self.alpha_eo = alpha_eo
        self.alpha_max = alpha_max
        self.warmup_epochs = warmup_epochs

    def get_alpha(self, epoch):
        """Get current fairness weight based on curriculum schedule."""
        if self.warmup_epochs <= 0:
            return self.alpha_max
        return self.alpha_max * min(1.0, epoch / self.warmup_epochs)

    def compute_sp_loss(self, output, sensitive, mask):
        """
        Statistical Parity regularization.

        Penalizes difference in prediction distributions between sensitive groups.

        Parameters
        ----------
        output : torch.Tensor
            Model output logits [N, C] or [N].
        sensitive : torch.Tensor
            Binary sensitive attributes [N].
        mask : torch.Tensor
            Boolean mask for training nodes.

        Returns
        -------
        torch.Tensor
            SP loss (scalar).
        """
        output = output[mask]
        sens = sensitive[mask]

        mask_0 = (sens == 0)
        mask_1 = (sens == 1)

        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return torch.tensor(0.0, device=output.device)

        if output.dim() == 1:
            # Binary classification with single logit
            probs = torch.sigmoid(output)
            mean_0 = probs[mask_0].mean()
            mean_1 = probs[mask_1].mean()
            return (mean_0 - mean_1).abs()
        else:
            # Multi-class: compare softmax distributions
            probs = F.softmax(output, dim=1)
            mean_0 = probs[mask_0].mean(dim=0)  # [C]
            mean_1 = probs[mask_1].mean(dim=0)  # [C]
            return (mean_0 - mean_1).abs().sum()

    def compute_eo_loss(self, output, sensitive, labels, mask):
        """
        Equal Opportunity regularization.

        Penalizes difference in true positive rates between sensitive groups.
        EO = |P(ŷ=1|y=1,s=0) - P(ŷ=1|y=1,s=1)|

        Parameters
        ----------
        output : torch.Tensor
            Model output logits [N, C] or [N].
        sensitive : torch.Tensor
            Binary sensitive attributes [N].
        labels : torch.Tensor
            Ground truth labels [N].
        mask : torch.Tensor
            Boolean mask for training nodes.

        Returns
        -------
        torch.Tensor
            EO loss (scalar).
        """
        output = output[mask]
        sens = sensitive[mask]
        labs = labels[mask]

        # Focus on positive class (y=1)
        pos_mask = (labs == 1)
        if pos_mask.sum() < 2:
            return torch.tensor(0.0, device=output.device)

        mask_0_pos = pos_mask & (sens == 0)
        mask_1_pos = pos_mask & (sens == 1)

        if mask_0_pos.sum() == 0 or mask_1_pos.sum() == 0:
            return torch.tensor(0.0, device=output.device)

        if output.dim() == 1:
            probs = torch.sigmoid(output)
            tpr_0 = probs[mask_0_pos].mean()
            tpr_1 = probs[mask_1_pos].mean()
        else:
            probs = F.softmax(output, dim=1)[:, 1]  # P(ŷ=1)
            tpr_0 = probs[mask_0_pos].mean()
            tpr_1 = probs[mask_1_pos].mean()

        return (tpr_0 - tpr_1).abs()

    def compute_structure_loss(self, adj, sensitive):
        """
        Structure-level fairness regularization.

        Penalizes imbalance in average edge weight between cross-group
        and same-group connections in the learned graph.

        Parameters
        ----------
        adj : torch.Tensor or tuple
            Learned adjacency [N,N], or edge-level tuple (indices, values, shape).
        sensitive : torch.Tensor
            Binary sensitive attributes [N].

        Returns
        -------
        torch.Tensor
            Structure fairness loss (scalar).
        """
        # Edge-level tuple from SparseGraphDiffusion
        if isinstance(adj, tuple):
            indices, values, shape = adj
            src, dst = indices[0], indices[1]

            s = sensitive.float()
            is_cross = (s[src] - s[dst]).abs()
            is_same = 1.0 - is_cross

            val_max = values.max().clamp(min=1e-8)
            val_norm = values / val_max

            cross_sum = (val_norm * is_cross).sum()
            same_sum = (val_norm * is_same).sum()
            n_cross = is_cross.sum().clamp(min=1)
            n_same = is_same.sum().clamp(min=1)

            return (cross_sum / n_cross - same_sum / n_same).abs()

        if adj.is_sparse:
            # Sparse path: compute structure loss from edge indices
            adj = adj.coalesce()
            indices = adj.indices()
            values = adj.values()
            src, dst = indices[0], indices[1]

            s = sensitive.float()
            is_cross = (s[src] - s[dst]).abs()  # 1 if cross-group
            is_same = 1.0 - is_cross

            val_max = values.max().clamp(min=1e-8)
            val_norm = values / val_max

            cross_sum = (val_norm * is_cross).sum()
            same_sum = (val_norm * is_same).sum()
            n_cross = is_cross.sum().clamp(min=1)
            n_same = is_same.sum().clamp(min=1)

            return (cross_sum / n_cross - same_sum / n_same).abs()

        # Dense path
        adj_max = adj.max().clamp(min=1e-8)
        adj_norm = adj / adj_max

        s = sensitive.float()
        cross_mask = (s.unsqueeze(0) - s.unsqueeze(1)).abs()
        same_mask = 1.0 - cross_mask

        edge_mask = (adj_norm > 0).float()
        cross_edges = adj_norm * cross_mask * edge_mask
        same_edges = adj_norm * same_mask * edge_mask

        n_cross = (cross_mask * edge_mask).sum().clamp(min=1)
        n_same = (same_mask * edge_mask).sum().clamp(min=1)

        avg_cross = cross_edges.sum() / n_cross
        avg_same = same_edges.sum() / n_same

        return (avg_cross - avg_same).abs()

    def forward(self, output, adj, sensitive, mask, epoch, labels=None):
        """
        Compute combined fairness loss.

        Parameters
        ----------
        output : torch.Tensor
            Model output logits.
        adj : torch.Tensor
            Learned adjacency matrix.
        sensitive : torch.Tensor
            Binary sensitive attributes.
        mask : torch.Tensor
            Training mask.
        epoch : int
            Current epoch (for curriculum scheduling).
        labels : torch.Tensor, optional
            Ground truth labels (required for EO loss).

        Returns
        -------
        torch.Tensor
            Weighted fairness loss.
        """
        alpha = self.get_alpha(epoch)

        loss_sp = self.compute_sp_loss(output, sensitive, mask)

        if adj is not None:
            loss_struct = self.compute_structure_loss(adj, sensitive)
        else:
            loss_struct = torch.tensor(0.0, device=output.device)

        loss_fair = self.alpha_sp * loss_sp + self.alpha_struct * loss_struct

        # EO loss (optional, enabled by alpha_eo > 0)
        if self.alpha_eo > 0 and labels is not None:
            loss_eo = self.compute_eo_loss(output, sensitive, labels, mask)
            loss_fair = loss_fair + self.alpha_eo * loss_eo

        return alpha * loss_fair
