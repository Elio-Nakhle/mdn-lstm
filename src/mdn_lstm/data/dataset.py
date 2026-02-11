"""Dataset classes and data loading utilities for MDN-LSTM."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataStats:
    """Statistics for data normalization."""

    mean: float
    std: float

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using stored statistics."""
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data using stored statistics."""
        return data * self.std + self.mean

    def save(self, path: Path) -> None:
        """Save statistics to file."""
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "DataStats":
        """Load statistics from file."""
        data = np.load(path)
        return cls(mean=float(data["mean"]), std=float(data["std"]))


class SequenceDataset(Dataset):
    """PyTorch Dataset for MDN-LSTM sequence data.

    Args:
        X: Input features of shape (n_samples, n_input).
        y: Target values of shape (n_samples, n_output).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_csv_data(path: Path, columns: list | None = None) -> tuple[np.ndarray, DataStats]:
    """Load and preprocess data from CSV file.

    This function handles the specific data format from the original notebook:
    - Reads specified columns (or defaults to specific lottery-like columns)
    - Parses and combines values into sequences
    - Removes invalid entries

    Args:
        path: Path to the CSV file.
        columns: List of column names to use (optional).

    Returns:
        Tuple of (data, stats):
            - data: Processed numpy array
            - stats: DataStats object with mean and std
    """
    df = pd.read_csv(path)
    df = df.dropna()

    if columns:
        df = df[columns]
    else:
        # Default columns from original notebook
        default_cols = ["1-2-7-21-27", "8-12", "44307952"]
        if all(col in df.columns for col in default_cols):
            df = df[default_cols]

    # Parse the data (specific to original format)
    processed = []
    for item in df.values:
        item_string = str(item[0]) + "-" + str(item[1]) + "-" + str(int(item[2] / 1000000))
        values = [int(number) for number in item_string.split("-")]
        processed.append(values)

    data = np.array(processed)

    # Remove rows where the last element is zero
    data = data[data[:, -1] != 0]

    # Calculate statistics and normalize
    stats = DataStats(mean=float(np.mean(data)), std=float(np.std(data)))
    normalized_data = stats.normalize(data)

    return normalized_data, stats


def load_generic_csv_data(
    path: Path, input_cols: list, normalize: bool = True
) -> tuple[np.ndarray, DataStats | None]:
    """Load generic CSV data with specified columns.

    Args:
        path: Path to the CSV file.
        input_cols: List of column names to use.
        normalize: Whether to normalize the data.

    Returns:
        Tuple of (data, stats) where stats is None if normalize=False.
    """
    df = pd.read_csv(path)
    df = df.dropna()
    data = df[input_cols].values.astype(np.float32)

    if normalize:
        stats = DataStats(mean=float(np.mean(data)), std=float(np.std(data)))
        normalized_data = stats.normalize(data)
        return normalized_data, stats
    else:
        return data, None


def prepare_sequences(
    data: np.ndarray, n_input: int = 8, n_output: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare input-output sequences for training.

    Creates sequences where X[i] is used to predict y[i] = data[i+1][:-1].

    Args:
        data: Normalized data array.
        n_input: Number of input features.
        n_output: Number of output features.

    Returns:
        Tuple of (X, y) arrays ready for training.
    """
    X = np.array(data[:-1]).astype(np.float32)
    y = np.array([row[:-1] for row in data[1:]]).astype(np.float32)

    # Reshape to expected dimensions
    X = X.reshape(X.shape[0], n_input)
    y = y.reshape(y.shape[0], n_output)

    return X, y


def train_val_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and validation sets.

    Args:
        X: Input features.
        y: Target values.
        train_ratio: Fraction of data to use for training.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    train_size = int(train_ratio * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    return X_train, X_val, y_train, y_val


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation.

    Args:
        X_train: Training input features.
        y_train: Training targets.
        X_val: Validation input features.
        y_val: Validation targets.
        batch_size: Batch size for DataLoaders.
        shuffle: Whether to shuffle training data.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
