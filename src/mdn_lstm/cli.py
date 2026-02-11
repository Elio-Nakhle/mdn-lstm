"""Command-line interface for MDN-LSTM."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.dataset import DataStats, load_csv_data, prepare_sequences, train_val_split
from .inference.predictor import Predictor
from .models.mdn import MDNLSTM
from .training.trainer import Trainer, load_model, save_model


def train_command(args: argparse.Namespace) -> int:
    """Execute training command."""
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Override with CLI arguments
    if args.epochs:
        config.training.epochs = args.epochs

    data_path = Path(args.data) if args.data else config.data.data_path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print(f"Loading data from {data_path}...")
    try:
        data, stats = load_csv_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Save data statistics
    stats.save(output_dir / "data_stats.npz")

    # Prepare sequences
    X, y = prepare_sequences(data, config.model.n_input, config.model.n_output)
    X_train, X_val, y_train, y_val = train_val_split(X, y, config.training.train_split)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create model
    model = MDNLSTM(
        n_input=config.model.n_input,
        n_hidden=config.model.n_hidden,
        n_output=config.model.n_output,
        n_gaussians=config.model.n_gaussians,
        num_lstm_layers=config.model.num_lstm_layers,
        bidirectional=config.model.bidirectional,
    )

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = Trainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_scheduler=config.training.use_scheduler,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_factor=config.training.scheduler_factor,
        l2_lambda=config.training.l2_lambda,
        device=device,
    )

    # Train
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.training.epochs,
        early_stopping=config.training.early_stopping,
        patience=config.training.early_stopping_patience,
        delta=config.training.early_stopping_delta,
        checkpoint_dir=output_dir / "checkpoints",
        verbose=not args.quiet,
    )

    # Save final model
    save_model(model, output_dir / "mdn_lstm_model.pt")
    print(f"Model saved to {output_dir / 'mdn_lstm_model.pt'}")

    # Save config used
    config.to_yaml(output_dir / "config_used.yaml")

    return 0


def validate_command(args: argparse.Namespace) -> int:
    """Execute validation command."""
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    data_path = Path(args.data) if args.data else config.data.data_path

    # Load data
    print(f"Loading data from {data_path}...")
    data, stats = load_csv_data(data_path)
    X, y = prepare_sequences(data, config.model.n_input, config.model.n_output)
    _, X_val, _, y_val = train_val_split(X, y, config.training.train_split)

    # Load model
    model = MDNLSTM(
        n_input=config.model.n_input,
        n_hidden=config.model.n_hidden,
        n_output=config.model.n_output,
        n_gaussians=config.model.n_gaussians,
        num_lstm_layers=config.model.num_lstm_layers,
        bidirectional=config.model.bidirectional,
    )
    model = load_model(model, model_path)

    # Create trainer just for validation
    trainer = Trainer(model=model, l2_lambda=config.training.l2_lambda)

    # Validate
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    val_loss = trainer.validate(X_val_tensor, y_val_tensor)

    print(f"Validation Loss: {val_loss:.6f}")

    return 0


def infer_command(args: argparse.Namespace) -> int:
    """Execute inference command."""
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    # Parse input
    try:
        input_values = [float(x.strip()) for x in args.input.split(",")]
    except ValueError as e:
        print(f"Invalid input format: {e}")
        return 1

    if len(input_values) != config.model.n_input:
        print(f"Expected {config.model.n_input} input values, got {len(input_values)}")
        return 1

    # Load data stats if available
    stats = None
    stats_path = Path(args.stats_path) if args.stats_path else model_path.parent / "data_stats.npz"
    if stats_path.exists():
        stats = DataStats.load(stats_path)

    # Create predictor
    predictor = Predictor.from_checkpoint(
        model_class=MDNLSTM,
        checkpoint_path=model_path,
        stats_path=stats_path if stats_path.exists() else None,
        n_input=config.model.n_input,
        n_hidden=config.model.n_hidden,
        n_output=config.model.n_output,
        n_gaussians=config.model.n_gaussians,
        num_lstm_layers=config.model.num_lstm_layers,
        bidirectional=config.model.bidirectional,
    )

    # Make prediction
    input_array = np.array(input_values)
    mu, sigma, pi = predictor.predict(
        input_array,
        normalize_input=stats is not None,
        denormalize_output=stats is not None,
    )

    print("\n--- Prediction Results ---")
    print(f"Input: {input_values}")
    print(f"Predicted mean: {mu}")
    print(f"Predicted std: {sigma}")
    print(f"Mixture weights: {pi}")

    if args.n_samples > 1:
        samples = predictor.sample(
            input_array,
            n_samples=args.n_samples,
            normalize_input=stats is not None,
            denormalize_output=stats is not None,
        )
        print(f"\n{args.n_samples} samples from distribution:")
        for i, sample in enumerate(samples[0]):
            print(f"  Sample {i + 1}: {np.round(sample, 2)}")

    return 0


def init_config_command(args: argparse.Namespace) -> int:
    """Generate default configuration file."""
    config = Config()
    output_path = Path(args.output)
    config.to_yaml(output_path)
    print(f"Default configuration saved to {output_path}")
    return 0


def export_command(args: argparse.Namespace) -> int:
    """Export model to ONNX format."""
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    # Load model
    model = MDNLSTM(
        n_input=config.model.n_input,
        n_hidden=config.model.n_hidden,
        n_output=config.model.n_output,
        n_gaussians=config.model.n_gaussians,
        num_lstm_layers=config.model.num_lstm_layers,
        bidirectional=config.model.bidirectional,
    )
    model = load_model(model, model_path)

    # Export to ONNX
    dummy_input = torch.randn(1, config.model.n_input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pi", "sigma", "mu"],
        dynamic_axes={"input": {0: "batch_size"}},
    )

    print(f"Model exported to {output_path}")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MDN-LSTM: Mixed Density Network with LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--data", type=str, help="Path to training data CSV")
    train_parser.add_argument("--output", type=str, default="models", help="Output directory")
    train_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate the model")
    val_parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    val_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    val_parser.add_argument("--data", type=str, help="Path to validation data CSV")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    infer_parser.add_argument(
        "--input", type=str, required=True, help="Comma-separated input values"
    )
    infer_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    infer_parser.add_argument("--stats-path", type=str, help="Path to data statistics")
    infer_parser.add_argument("--n-samples", type=int, default=1, help="Number of samples")

    # Init config command
    init_parser = subparsers.add_parser("init-config", help="Generate default config")
    init_parser.add_argument("--output", type=str, default="config.yaml", help="Output path")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    export_parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    export_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        return train_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "infer":
        return infer_command(args)
    elif args.command == "init-config":
        return init_config_command(args)
    elif args.command == "export":
        return export_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
