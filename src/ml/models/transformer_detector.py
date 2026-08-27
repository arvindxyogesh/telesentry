"""Self-attention reconstruction detector for multivariate telemetry windows.

Design is inspired by the reconstruction + self-attention line of anomaly
detection research that has become a standard comparison point since Xu et
al.'s Anomaly Transformer (ICLR 2022) and Tuli et al.'s TranAD (VLDB 2022):
a Transformer encoder reconstructs each window, and reconstruction error is
combined with a measure of how *concentrated* each timestep's attention is.

This module implements a simplified, single-branch version of that idea
(plain reconstruction + attention-concentration penalty) rather than the
original two-branch adversarial "association discrepancy" minimax training,
which is out of scope here -- the goal is to test whether attention-derived
structure adds a genuinely useful signal on top of plain reconstruction for
this domain, not to reproduce the original papers' exact objective.

Intuition for the attention term: over a normal window, a point is easy to
explain by *several* nearby points (broad, smooth attention). An anomalous
point tends to be either unlike anything nearby (attention has nowhere good
to go, so it collapses onto itself) or part of a short, unfamiliar sub-
pattern -- both cases produce lower-entropy ("peakier") attention than the
model's typical behavior on normal data.
"""

import math

import torch
from torch import nn


class _SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights  # attn_weights: (batch, window, window), rows sum to 1


class TransformerAnomalyDetector(nn.Module):
    def __init__(
        self,
        num_features: int,
        window: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        self.blocks = nn.ModuleList([_SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x: torch.Tensor):
        """x: (batch, window, num_features).
        Returns (reconstruction, attn_weights_last_layer).
        """
        h = self.input_proj(x) + self.pos_embedding
        attn_weights = None
        for block in self.blocks:
            h, attn_weights = block(h)
        recon = self.output_proj(h)
        return recon, attn_weights

    @staticmethod
    def attention_concentration(attn_weights: torch.Tensor) -> torch.Tensor:
        """Negative normalized entropy of the *last* query position's
        attention row: 0 = uniform/maximally spread out, 1 = fully collapsed
        onto a single key. Higher = more anomalous under the association
        hypothesis above. Shape (batch,).
        """
        last_row = attn_weights[:, -1, :]  # (batch, window)
        eps = 1e-8
        entropy = -torch.sum(last_row * torch.log(last_row + eps), dim=-1)
        max_entropy = math.log(attn_weights.shape[-1])
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy

    def anomaly_components(self, x: torch.Tensor):
        """Returns (reconstruction_error, attention_concentration), both
        shape (batch,), evaluated at the window's last timestep.
        """
        recon, attn_weights = self.forward(x)
        recon_error = torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=-1)
        concentration = self.attention_concentration(attn_weights)
        return recon_error, concentration


__all__ = ["TransformerAnomalyDetector"]
