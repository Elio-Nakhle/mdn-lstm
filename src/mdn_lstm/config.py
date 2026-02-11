"""Configuration dataclasses for MDN-LSTM."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    n_input: int = 8
    n_output: int = 7
    n_hidden: int = 2
    n_gaussians: int = 3
    num_lstm_layers: int = 1
    bidirectional: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 84000
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    batch_size: int = 32
    train_split: float = 0.8
    use_scheduler: bool = False
    scheduler_patience: int = 100
    scheduler_factor: float = 0.5
    early_stopping: bool = True
    early_stopping_patience: int = 2000
    early_stopping_delta: float = 0.0005
    l2_lambda: float = 0.001


@dataclass
class DataConfig:
    """Data configuration."""

    data_path: Path = field(default_factory=lambda: Path("data.csv"))
    model_save_path: Path = field(default_factory=lambda: Path("models"))
    checkpoint_path: Path = field(default_factory=lambda: Path("checkpoints"))


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        model_cfg = ModelConfig(**data.get("model", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        data_cfg_dict = data.get("data", {})
        # Convert string paths to Path objects
        for key in ["data_path", "model_save_path", "checkpoint_path"]:
            if key in data_cfg_dict:
                data_cfg_dict[key] = Path(data_cfg_dict[key])
        data_cfg = DataConfig(**data_cfg_dict)

        return cls(model=model_cfg, training=training_cfg, data=data_cfg)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": {
                "n_input": self.model.n_input,
                "n_output": self.model.n_output,
                "n_hidden": self.model.n_hidden,
                "n_gaussians": self.model.n_gaussians,
                "num_lstm_layers": self.model.num_lstm_layers,
                "bidirectional": self.model.bidirectional,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "batch_size": self.training.batch_size,
                "train_split": self.training.train_split,
                "use_scheduler": self.training.use_scheduler,
                "scheduler_patience": self.training.scheduler_patience,
                "scheduler_factor": self.training.scheduler_factor,
                "early_stopping": self.training.early_stopping,
                "early_stopping_patience": self.training.early_stopping_patience,
                "early_stopping_delta": self.training.early_stopping_delta,
                "l2_lambda": self.training.l2_lambda,
            },
            "data": {
                "data_path": str(self.data.data_path),
                "model_save_path": str(self.data.model_save_path),
                "checkpoint_path": str(self.data.checkpoint_path),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
