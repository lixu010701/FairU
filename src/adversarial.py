"""
Adversarial debiasing for FairUnGSL v5.

Uses Gradient Reversal Layer (GRL) to train the GCN encoder to produce
representations that cannot predict sensitive attributes.

The GRL reverses gradients during backprop:
- Forward: h → discriminator predicts sensitive attribute (normal)
- Backward: gradient is NEGATED → encoder learns to NOT encode sensitive info

This is backbone-agnostic: it only requires the GCN hidden representations,
regardless of which backbone (GRCN/IDGL/PROGNN/PROSE) produced the graph.

Reference: FairGNN (Dai & Wang, 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL).

    Forward: identity.
    Backward: negate gradient and scale by alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SensitiveDiscriminator(nn.Module):
    """Discriminator that predicts sensitive attribute from node representations.

    Architecture: h → GRL → MLP → P(sensitive=1)

    The GRL ensures that when this loss is minimized, the encoder is pushed
    to produce representations that CANNOT predict sensitive attributes.

    Parameters
    ----------
    n_hidden : int
        Dimension of input hidden representations.
    n_disc_hidden : int
        Hidden dimension of discriminator MLP.
    dropout : float
        Dropout rate in discriminator.
    """

    def __init__(self, n_hidden, n_disc_hidden=None, dropout=0.3):
        super().__init__()
        if n_disc_hidden is None:
            n_disc_hidden = max(n_hidden // 2, 16)

        self.net = nn.Sequential(
            nn.Linear(n_hidden, n_disc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_disc_hidden, 1),
        )

    def forward(self, h, alpha=1.0):
        """Forward pass with gradient reversal.

        Parameters
        ----------
        h : torch.Tensor [N, n_hidden]
            Node hidden representations from GCN encoder.
        alpha : float
            Gradient reversal strength (curriculum: start small, increase).

        Returns
        -------
        torch.Tensor [N]
            Predicted probability of sensitive=1 (logits).
        """
        h_rev = GradientReversalFunction.apply(h, alpha)
        return self.net(h_rev).squeeze(-1)


def compute_adv_loss(discriminator, hidden, sensitive, mask, alpha=1.0):
    """Compute adversarial debiasing loss.

    Parameters
    ----------
    discriminator : SensitiveDiscriminator
    hidden : torch.Tensor [N, n_hidden]
        GCN hidden representations.
    sensitive : torch.Tensor [N]
        Binary sensitive attributes.
    mask : torch.Tensor [N]
        Training mask.
    alpha : float
        GRL strength (curriculum scheduled).

    Returns
    -------
    torch.Tensor
        Binary cross-entropy loss for sensitive prediction.
        Minimizing this + GRL = encoder learns fair representations.
    """
    sens_logits = discriminator(hidden, alpha=alpha)
    loss = F.binary_cross_entropy_with_logits(
        sens_logits[mask], sensitive[mask].float()
    )
    return loss
