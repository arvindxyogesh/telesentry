import torch

from src.ml.models.lstm_autoencoder import LSTMAutoencoder
from src.ml.models.transformer_detector import TransformerAnomalyDetector

BATCH, WINDOW, NUM_FEATURES = 6, 15, 7


def test_lstm_autoencoder_reconstruction_shape():
    model = LSTMAutoencoder(num_features=NUM_FEATURES, hidden_size=16)
    x = torch.randn(BATCH, WINDOW, NUM_FEATURES)
    recon = model(x)
    assert recon.shape == x.shape


def test_lstm_autoencoder_reconstruction_error_shape_and_nonnegative():
    model = LSTMAutoencoder(num_features=NUM_FEATURES, hidden_size=16)
    x = torch.randn(BATCH, WINDOW, NUM_FEATURES)
    err = model.reconstruction_error(x)
    assert err.shape == (BATCH,)
    assert (err >= 0).all()


def test_transformer_detector_shapes():
    model = TransformerAnomalyDetector(num_features=NUM_FEATURES, window=WINDOW, d_model=16, nhead=2, num_layers=2)
    model.eval()  # disable attention dropout so weight rows sum exactly to 1
    x = torch.randn(BATCH, WINDOW, NUM_FEATURES)
    recon, attn = model(x)
    assert recon.shape == x.shape
    assert attn.shape == (BATCH, WINDOW, WINDOW)
    # Attention weights are a valid distribution over keys for each query.
    assert torch.allclose(attn.sum(dim=-1), torch.ones(BATCH, WINDOW), atol=1e-4)


def test_transformer_anomaly_components():
    model = TransformerAnomalyDetector(num_features=NUM_FEATURES, window=WINDOW, d_model=16, nhead=2, num_layers=2)
    model.eval()
    x = torch.randn(BATCH, WINDOW, NUM_FEATURES)
    recon_err, concentration = model.anomaly_components(x)
    assert recon_err.shape == (BATCH,)
    assert concentration.shape == (BATCH,)
    assert (recon_err >= 0).all()
    # Concentration is a normalized-entropy-derived score in ~[0, 1].
    assert concentration.max().item() <= 1.01
    assert concentration.min().item() >= -0.1


def test_attention_concentration_uniform_is_zero():
    uniform = torch.full((2, WINDOW, WINDOW), 1.0 / WINDOW)
    concentration = TransformerAnomalyDetector.attention_concentration(uniform)
    assert torch.allclose(concentration, torch.zeros(2), atol=1e-4)


def test_attention_concentration_one_hot_is_one():
    one_hot = torch.zeros(1, WINDOW, WINDOW)
    one_hot[:, :, -1] = 1.0
    concentration = TransformerAnomalyDetector.attention_concentration(one_hot)
    assert torch.allclose(concentration, torch.ones(1), atol=1e-3)
