"""Inference and prediction utilities for MDN-LSTM models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import DataStats


class Predictor:
    """Predictor class for MDN-LSTM inference.

    Args:
        model: The trained MDN or MDN-LSTM model.
        data_stats: Data statistics for normalization/denormalization.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        data_stats: DataStats | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.data_stats = data_stats
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        checkpoint_path: Path,
        stats_path: Path | None = None,
        device: str = "cpu",
        **model_kwargs,
    ) -> "Predictor":
        """Create a Predictor from a saved checkpoint.

        Args:
            model_class: The model class (MDNLSTM or MDN).
            checkpoint_path: Path to the saved model checkpoint.
            stats_path: Path to saved data statistics (optional).
            device: Device to run inference on.
            **model_kwargs: Arguments to pass to model constructor.

        Returns:
            Initialized Predictor instance.
        """
        model = model_class(**model_kwargs)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        data_stats = None
        if stats_path and stats_path.exists():
            data_stats = DataStats.load(stats_path)

        return cls(model, data_stats, device)

    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray | torch.Tensor,
        normalize_input: bool = True,
        denormalize_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make a prediction and return the most likely output.

        Args:
            x: Input features of shape (n_input,) or (batch_size, n_input).
            normalize_input: Whether to normalize input using data_stats.
            denormalize_output: Whether to denormalize output using data_stats.

        Returns:
            Tuple of (mu, sigma, pi):
                - mu: Mean values of the most likely Gaussian.
                - sigma: Standard deviation of the most likely Gaussian.
                - pi: Mixing coefficients.
        """
        # Ensure numpy array
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # Normalize if requested
        if normalize_input and self.data_stats:
            x = self.data_stats.normalize(x)

        # Convert to tensor and add batch dimension if needed
        x_tensor = torch.FloatTensor(x)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        x_tensor = x_tensor.to(self.device)

        # Forward pass
        pi, sigma, mu = self.model(x_tensor)

        # Get most likely prediction
        _, mu_out, sigma_out = self.model.get_most_likely_prediction(pi, sigma, mu)

        # Denormalize if requested
        if denormalize_output and self.data_stats:
            mu_out = self.data_stats.denormalize(mu_out)

        return mu_out, sigma_out, pi.cpu().numpy()

    @torch.no_grad()
    def predict_distribution(
        self,
        x: np.ndarray | torch.Tensor,
        normalize_input: bool = True,
        denormalize_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the full mixture distribution parameters.

        Args:
            x: Input features.
            normalize_input: Whether to normalize input.
            denormalize_output: Whether to denormalize outputs.

        Returns:
            Tuple of (pi, sigma, mu) arrays.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if normalize_input and self.data_stats:
            x = self.data_stats.normalize(x)

        x_tensor = torch.FloatTensor(x)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        x_tensor = x_tensor.to(self.device)

        pi, sigma, mu = self.model(x_tensor)

        pi_np = pi.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        mu_np = mu.cpu().numpy()

        if denormalize_output and self.data_stats:
            # Denormalize all means
            mu_np = mu_np * self.data_stats.std + self.data_stats.mean
            sigma_np = sigma_np * self.data_stats.std

        return pi_np, sigma_np, mu_np

    @torch.no_grad()
    def sample(
        self,
        x: np.ndarray | torch.Tensor,
        n_samples: int = 10,
        normalize_input: bool = True,
        denormalize_output: bool = True,
    ) -> np.ndarray:
        """Sample from the predicted distribution.

        Args:
            x: Input features.
            n_samples: Number of samples to draw.
            normalize_input: Whether to normalize input.
            denormalize_output: Whether to denormalize outputs.

        Returns:
            Samples of shape (batch_size, n_samples, n_output).
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if normalize_input and self.data_stats:
            x = self.data_stats.normalize(x)

        x_tensor = torch.FloatTensor(x)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        x_tensor = x_tensor.to(self.device)

        pi, sigma, mu = self.model(x_tensor)
        batch_size = pi.shape[0]
        valid_samples = [[] for _ in range(batch_size)]
        # Keep sampling until each batch has n_samples valid samples
        while any(len(v) < n_samples for v in valid_samples):
            needed = [n_samples - len(v) for v in valid_samples]
            max_needed = max(needed)
            # Sample more than needed to increase chance of filling
            extra = max(2, max_needed)
            samples = self.model.sample(pi, sigma, mu, extra)
            samples_np = samples.numpy()
            if denormalize_output and self.data_stats:
                samples_np = self.data_stats.denormalize(samples_np)
            # samples_np shape: (batch_size, extra, n_output)
            for batch_idx in range(batch_size):
                for s in samples_np[batch_idx]:
                    # Reject samples with any value less than 0.5 (including negatives)
                    if (s >= 0.5).all() and len(valid_samples[batch_idx]) < n_samples:
                        valid_samples[batch_idx].append(s)
        # Convert to numpy array
        result = np.array(valid_samples, dtype=samples_np.dtype)
        return result


def batch_predict(
    predictor: Predictor,
    X: np.ndarray,
    batch_size: int = 32,
    normalize_input: bool = True,
    denormalize_output: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions on a batch of inputs.

    Args:
        predictor: Predictor instance.
        X: Input features of shape (n_samples, n_input).
        batch_size: Batch size for processing.
        normalize_input: Whether to normalize inputs.
        denormalize_output: Whether to denormalize outputs.

    Returns:
        Tuple of (predictions, uncertainties).
    """
    n_samples = len(X)
    predictions = []
    uncertainties = []

    for i in range(0, n_samples, batch_size):
        batch = X[i : i + batch_size]
        mu, sigma, _ = predictor.predict(
            batch,
            normalize_input=normalize_input,
            denormalize_output=denormalize_output,
        )
        predictions.append(mu)
        uncertainties.append(sigma)

    return np.concatenate(predictions), np.concatenate(uncertainties)
