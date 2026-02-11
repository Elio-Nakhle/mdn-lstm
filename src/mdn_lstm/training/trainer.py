"""Training loop and utilities for MDN-LSTM models."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..models.mdn import mdn_loss


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


class Trainer:
    """Trainer class for MDN-LSTM models.

    Args:
        model: The MDN or MDN-LSTM model to train.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        use_scheduler: Whether to use learning rate scheduler.
        scheduler_patience: Patience for learning rate reduction.
        scheduler_factor: Factor for learning rate reduction.
        l2_lambda: L2 regularization coefficient in loss function.
        device: Device to train on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        use_scheduler: bool = False,
        scheduler_patience: int = 100,
        scheduler_factor: float = 0.5,
        l2_lambda: float = 0.001,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.l2_lambda = l2_lambda
        self.use_scheduler = use_scheduler

        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
            )

        self.metrics = TrainingMetrics()

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Train for one epoch.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            Training loss for this epoch.
        """
        self.model.train()

        # Shuffle data
        perm = torch.randperm(len(X))
        X_shuffled = X[perm].to(self.device)
        y_shuffled = y[perm].to(self.device)

        # Forward pass
        pi, sigma, mu = self.model(X_shuffled)

        # Compute loss
        loss = mdn_loss(pi, sigma, mu, y_shuffled, self.model, l2_lambda=self.l2_lambda)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.metrics.gradient_norms.append(total_norm)

        return loss.item()

    @torch.no_grad()
    def validate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Validate the model.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            Validation loss.
        """
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)

        pi, sigma, mu = self.model(X)
        loss = mdn_loss(pi, sigma, mu, y, self.model, l2_lambda=self.l2_lambda)

        return loss.item()

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        early_stopping: bool = True,
        patience: int = 2000,
        delta: float = 0.0005,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """Full training loop.

        Args:
            X_train: Training input features.
            y_train: Training targets.
            X_val: Validation input features.
            y_val: Validation targets.
            epochs: Maximum number of epochs.
            early_stopping: Whether to use early stopping.
            patience: Epochs to wait before early stopping.
            delta: Minimum improvement for early stopping.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Epochs between logging.
            verbose: Whether to print progress.

        Returns:
            TrainingMetrics with training history.
        """
        if verbose:
            print("--------- Starting Training ---------")

        X_train = torch.FloatTensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
        y_train = torch.FloatTensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
        X_val = torch.FloatTensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val
        y_val = torch.FloatTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training step
            train_loss = self.train_epoch(X_train, y_train)
            self.metrics.train_losses.append(train_loss)

            # Check for NaN
            if math.isnan(train_loss):
                if verbose:
                    print(f"NaN loss detected at epoch {epoch}. Stopping training.")
                break

            # Convergence check
            if train_loss < -0.7:
                if verbose:
                    print(f"Training converged at epoch {epoch} with loss {train_loss:.6f}")
                break

            # Validation (less frequently for efficiency)
            if epoch % log_interval == 0:
                val_loss = self.validate(X_val, y_val)
                self.metrics.val_losses.append(val_loss)

                if verbose:
                    grad_norm = (
                        self.metrics.gradient_norms[-1] if self.metrics.gradient_norms else 0
                    )
                    print(
                        f"Epoch {epoch}: "
                        f"Train Loss={train_loss:.6f}, "
                        f"Val Loss={val_loss:.6f}, "
                        f"Grad Norm={grad_norm:.4f}"
                    )

                # Save best model
                if val_loss < self.metrics.best_val_loss:
                    self.metrics.best_val_loss = val_loss
                    self.metrics.best_epoch = epoch

                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / "best_model.pt")

            # Learning rate scheduling
            if self.use_scheduler and self.metrics.val_losses:
                self.scheduler.step(self.metrics.val_losses[-1])

            # Early stopping
            if early_stopping and self.metrics.val_losses:
                if self.metrics.val_losses[-1] < self.metrics.best_val_loss - delta:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if verbose:
            print(f"--------- Training Complete ---------")
            print(
                f"Best validation loss: {self.metrics.best_val_loss:.6f} at epoch {self.metrics.best_epoch}"
            )

        return self.metrics

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": {
                    "train_losses": self.metrics.train_losses,
                    "val_losses": self.metrics.val_losses,
                    "best_val_loss": self.metrics.best_val_loss,
                    "best_epoch": self.metrics.best_epoch,
                },
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            self.metrics.train_losses = metrics.get("train_losses", [])
            self.metrics.val_losses = metrics.get("val_losses", [])
            self.metrics.best_val_loss = metrics.get("best_val_loss", float("inf"))
            self.metrics.best_epoch = metrics.get("best_epoch", 0)


def save_model(model: nn.Module, path: Path) -> None:
    """Save only the model state dict.

    Args:
        model: The model to save.
        path: Path to save the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: Path, device: str = "cpu") -> nn.Module:
    """Load model state dict.

    Args:
        model: The model to load weights into.
        path: Path to the saved model.
        device: Device to load the model onto.

    Returns:
        Model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
