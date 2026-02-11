"""Tests for MDN-LSTM models."""

import numpy as np
import pytest
import torch

from mdn_lstm.models.mdn import MDNLSTM, MDN, mdn_loss, gaussian_pdf


class TestMDNLSTM:
    """Tests for the MDNLSTM model."""

    def test_model_creation(self):
        """Test model can be instantiated."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )
        assert model is not None
        assert model.n_input == 8
        assert model.n_output == 7

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        batch_size = 4
        x = torch.randn(batch_size, 8)

        pi, sigma, mu = model(x)

        assert pi.shape == (batch_size, 3)
        assert sigma.shape == (batch_size, 3)
        assert mu.shape == (batch_size, 3 * 7)

    def test_forward_pass_with_sequence(self):
        """Test forward pass with sequence input."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        batch_size = 4
        seq_len = 5
        x = torch.randn(batch_size, seq_len, 8)

        pi, sigma, mu = model(x)

        assert pi.shape == (batch_size, 3)
        assert sigma.shape == (batch_size, 3)
        assert mu.shape == (batch_size, 3 * 7)

    def test_pi_sums_to_one(self):
        """Test that mixture weights sum to 1."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(4, 8)
        pi, _, _ = model(x)

        # Check that pi sums to 1 for each sample
        sums = pi.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_sigma_positive(self):
        """Test that sigma values are positive."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(4, 8)
        _, sigma, _ = model(x)

        assert (sigma > 0).all()

    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM option."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
            bidirectional=True,
        )

        x = torch.randn(4, 8)
        pi, sigma, mu = model(x)

        assert pi.shape == (4, 3)


class TestMDN:
    """Tests for the standard MDN model."""

    def test_model_creation(self):
        """Test model can be instantiated."""
        model = MDN(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = MDN(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(4, 8)
        pi, sigma, mu = model(x)

        assert pi.shape == (4, 3)
        assert sigma.shape == (4, 3)
        assert mu.shape == (4, 21)


class TestMDNLoss:
    """Tests for MDN loss function."""

    def test_loss_computation(self):
        """Test loss can be computed."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(4, 8)
        y = torch.randn(4, 7)

        pi, sigma, mu = model(x)
        loss = mdn_loss(pi, sigma, mu, y, model)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_loss_gradient_flow(self):
        """Test gradients flow through the loss."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(4, 8)
        y = torch.randn(4, 7)

        pi, sigma, mu = model(x)
        loss = mdn_loss(pi, sigma, mu, y, model)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestSampling:
    """Tests for sampling from MDN."""

    def test_sample_shape(self):
        """Test sample produces correct shape."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(2, 8)
        pi, sigma, mu = model(x)

        samples = model.sample(pi, sigma, mu, n_samples=10)

        assert samples.shape == (2, 10, 7)

    def test_get_most_likely_prediction(self):
        """Test most likely prediction extraction."""
        model = MDNLSTM(
            n_input=8,
            n_hidden=16,
            n_output=7,
            n_gaussians=3,
        )

        x = torch.randn(1, 8)
        pi, sigma, mu = model(x)

        idx, mu_out, sigma_out = model.get_most_likely_prediction(pi, sigma, mu)

        assert len(mu_out) == 7
        assert isinstance(sigma_out, np.ndarray)
