"""Sequence-to-sequence LSTM autoencoder for unsupervised anomaly detection.

Trained only on windows drawn from normal telemetry to reconstruct its input
window; at inference time the per-timestep reconstruction error is the
anomaly score. This is the classic recurrent baseline for multivariate time
series anomaly detection (Malhotra et al., 2016, "LSTM-based Encoder-Decoder
for Multi-sensor Anomaly Detection") and remains a standard comparison point
in current (2023-2025) deep anomaly detection papers.
"""

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, num_features) -> reconstruction of the same shape."""
        batch, window, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]  # (batch, hidden_size): final encoder hidden state

        decoder_input = latent.unsqueeze(1).repeat(1, window, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_proj(decoded)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error at the *last* timestep of the
        window -- the causal "anomaly score right now" for a trailing window.
        Returns shape (batch,).
        """
        recon = self.forward(x)
        return torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=-1)


__all__ = ["LSTMAutoencoder"]
