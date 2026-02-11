"""Mixed Density Network with LSTM implementation."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Normalization factor for Gaussians
ONE_DIV_SQRT_TWO_PI = 1.0 / np.sqrt(2.0 * np.pi)


class MDNLSTM(nn.Module):
    """Mixed Density Network with LSTM for sequence prediction.

    This model combines an LSTM layer for temporal feature extraction
    with a Mixture Density Network output layer for probabilistic predictions.

    Args:
        n_input: Number of input features.
        n_hidden: Hidden size for LSTM layer.
        n_output: Number of output dimensions.
        n_gaussians: Number of Gaussian components in the mixture.
        num_lstm_layers: Number of LSTM layers (default: 1).
        bidirectional: Whether to use bidirectional LSTM (default: False).
        dropout: Dropout probability for LSTM (default: 0.0).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_gaussians: int = 3,
        num_lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_gaussians = n_gaussians
        self.bidirectional = bidirectional

        # LSTM layer for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=n_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Calculate actual hidden dimension (doubled if bidirectional)
        lstm_output_size = n_hidden * (2 if bidirectional else 1)

        # MDN output layers
        self.z_pi = nn.Linear(lstm_output_size, n_gaussians)  # Mixture weights
        self.z_sigma = nn.Linear(lstm_output_size, n_gaussians)  # Std deviations
        self.z_mu = nn.Linear(lstm_output_size, n_gaussians * n_output)  # Means

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_input) or
               (batch_size, sequence_length, n_input).
            hidden: Optional tuple of (h_n, c_n) hidden states.

        Returns:
            Tuple of (pi, sigma, mu):
                - pi: Mixture weights of shape (batch_size, n_gaussians)
                - sigma: Standard deviations of shape (batch_size, n_gaussians)
                - mu: Means of shape (batch_size, n_gaussians * n_output)
        """
        # Handle 2D input (batch_size, n_input) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, n_input)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x, hidden)

        # Extract output from the last time step
        z_h = lstm_out[:, -1, :]

        # Compute mixture parameters
        pi = nn.functional.softmax(self.z_pi(z_h), dim=-1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)

        return pi, sigma, mu

    def sample(
        self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, n_samples: int = 10
    ) -> torch.Tensor:
        """Sample from the mixture distribution.

        Args:
            pi: Mixture weights of shape (batch_size, n_gaussians).
            sigma: Standard deviations of shape (batch_size, n_gaussians).
            mu: Means of shape (batch_size, n_gaussians * n_output).
            n_samples: Number of samples per input.

        Returns:
            Samples of shape (batch_size, n_samples, n_output).
        """
        N, K = pi.shape
        T = self.n_output
        samples = torch.zeros(N, n_samples, T)

        for i in range(N):
            for j in range(n_samples):
                # Sample component from categorical distribution
                u = np.random.uniform()
                prob_sum = 0.0
                for k in range(K):
                    prob_sum += pi[i, k].item()
                    if u < prob_sum:
                        # Sample from the k-th Gaussian component
                        for t in range(T):
                            mean = mu[i, k * T + t].item()
                            std = sigma[i, k].item()
                            samples[i, j, t] = np.random.normal(mean, std)
                        break

        return samples

    def get_most_likely_prediction(
        self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most likely prediction (mode of the mixture).

        Args:
            pi: Mixture weights of shape (batch_size, n_gaussians).
            sigma: Standard deviations of shape (batch_size, n_gaussians).
            mu: Means of shape (batch_size, n_gaussians * n_output).

        Returns:
            Tuple of (component_idx, mu_out, sigma_out):
                - component_idx: Index of the most likely component
                - mu_out: Mean values for the most likely component
                - sigma_out: Standard deviation of the most likely component
        """
        _, imax = torch.max(pi, dim=1)
        imax_scalar = imax.item() if imax.ndim == 0 else imax[0].item()

        T = self.n_output
        mu_out = np.zeros(T)
        for t in range(T):
            mu_out[t] = mu[0, imax_scalar * T + t].detach().cpu().numpy()

        sigma_out = sigma[0, imax_scalar].detach().cpu().numpy()

        return np.array([imax_scalar]), mu_out, sigma_out


class MDN(nn.Module):
    """Standard Mixed Density Network (without LSTM).

    This is the original MDN implementation with a simple feedforward
    hidden layer instead of LSTM.

    Args:
        n_input: Number of input features.
        n_hidden: Hidden layer size.
        n_output: Number of output dimensions.
        n_gaussians: Number of Gaussian components in the mixture.
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int, n_gaussians: int = 3):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_gaussians = n_gaussians

        self.z_h = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians * n_output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_input).

        Returns:
            Tuple of (pi, sigma, mu).
        """
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), dim=-1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

    def get_most_likely_prediction(
        self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most likely prediction (mode of the mixture)."""
        _, imax = torch.max(pi, dim=1)
        imax_scalar = imax.item() if imax.ndim == 0 else imax[0].item()

        T = self.n_output
        mu_out = np.zeros(T)
        for t in range(T):
            mu_out[t] = mu[0, imax_scalar * T + t].detach().cpu().numpy()

        sigma_out = sigma[0, imax_scalar].detach().cpu().numpy()

        return np.array([imax_scalar]), mu_out, sigma_out


def gaussian_pdf(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian probability density.

    Args:
        y: Target values of shape (batch_size, n_output).
        mu: Mean values of shape (batch_size, n_output).
        sigma: Standard deviation (scalar per Gaussian).

    Returns:
        Probability density values.
    """
    sigma_inv = torch.reciprocal(sigma)
    result = torch.norm((y - mu), 2, 1) * sigma_inv
    result = -0.5 * (result * result)
    return (torch.exp(result) * sigma_inv) * ONE_DIV_SQRT_TWO_PI


def mdn_loss(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    epsilon: float = 1e-8,
    l2_lambda: float = 0.001,
) -> torch.Tensor:
    """Compute MDN negative log-likelihood loss with L2 regularization.

    Args:
        pi: Mixture weights of shape (batch_size, n_gaussians).
        sigma: Standard deviations of shape (batch_size, n_gaussians).
        mu: Means of shape (batch_size, n_gaussians * n_output).
        y: Target values of shape (batch_size, n_output).
        model: The model (for L2 regularization).
        epsilon: Small constant for numerical stability.
        l2_lambda: L2 regularization coefficient.

    Returns:
        Loss value (scalar tensor).
    """
    N, K = pi.shape
    _, KT = mu.shape
    _, T = y.shape
    losses = torch.zeros(N, device=pi.device)

    for k in range(K):
        likelihood = gaussian_pdf(y, mu[:, k * T : (k + 1) * T], sigma[:, k])
        losses += pi[:, k] * likelihood

    # Clamp for numerical stability
    losses = torch.clamp(losses, min=epsilon)
    nll_loss = torch.mean(-torch.log(losses))

    # L2 regularization
    l2_reg = torch.tensor(0.0, device=pi.device)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)

    return nll_loss + l2_lambda * l2_reg
