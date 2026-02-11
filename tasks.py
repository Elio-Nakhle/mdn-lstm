"""
Invoke tasks for MDN-LSTM project.

Usage:
    invoke train --config config.yaml
    invoke validate --model-path models/best_model.pt
    invoke infer --model-path models/best_model.pt --input "12,22,28,30,31,4,11,31"
    invoke export --model-path models/best_model.pt --output model.onnx
"""

from pathlib import Path

from invoke import task, Context


@task
def install(ctx: Context, dev: bool = True):
    """Install project dependencies using PDM.

    Args:
        dev: Include development dependencies.
    """
    cmd = "pdm install"
    if dev:
        cmd += " --dev"
    ctx.run(cmd)


@task
def train(
    ctx: Context,
    config: str = "config.yaml",
    epochs: int = None,
    data: str = None,
    output: str = "models",
    verbose: bool = True,
):
    """Train the MDN-LSTM model.

    Args:
        config: Path to configuration YAML file.
        epochs: Number of training epochs (overrides config).
        data: Path to training data CSV (overrides config).
        output: Output directory for models.
        verbose: Print training progress.
    """
    import sys

    args = ["python", "-m", "mdn_lstm.cli", "train"]
    args.extend(["--config", config])

    if epochs:
        args.extend(["--epochs", str(epochs)])
    if data:
        args.extend(["--data", data])
    args.extend(["--output", output])
    if not verbose:
        args.append("--quiet")

    ctx.run(" ".join(args))


@task
def validate(
    ctx: Context,
    model_path: str,
    config: str = "config.yaml",
    data: str = None,
):
    """Validate the trained model.

    Args:
        model_path: Path to the trained model checkpoint.
        config: Path to configuration YAML file.
        data: Path to validation data CSV (overrides config).
    """
    args = ["python", "-m", "mdn_lstm.cli", "validate"]
    args.extend(["--model-path", model_path])
    args.extend(["--config", config])
    if data:
        args.extend(["--data", data])

    ctx.run(" ".join(args))


@task
def infer(
    ctx: Context,
    model_path: str,
    input_data: str,
    config: str = "config.yaml",
    stats_path: str = None,
    n_samples: int = 1,
):
    """Run inference with the trained model.

    Args:
        model_path: Path to the trained model checkpoint.
        input_data: Comma-separated input values.
        config: Path to configuration YAML file.
        stats_path: Path to data statistics file.
        n_samples: Number of samples to generate.
    """
    args = ["python", "-m", "mdn_lstm.cli", "infer"]
    args.extend(["--model-path", model_path])
    args.extend(["--input", f'"{input_data}"'])
    args.extend(["--config", config])
    if stats_path:
        args.extend(["--stats-path", stats_path])
    args.extend(["--n-samples", str(n_samples)])

    ctx.run(" ".join(args))


@task
def export_model(ctx: Context, model_path: str, output: str, config: str = "config.yaml"):
    """Export model to ONNX format.

    Args:
        model_path: Path to the trained model checkpoint.
        output: Output path for ONNX model.
        config: Path to configuration YAML file.
    """
    args = ["python", "-m", "mdn_lstm.cli", "export"]
    args.extend(["--model-path", model_path])
    args.extend(["--output", output])
    args.extend(["--config", config])

    ctx.run(" ".join(args))


@task
def test(ctx: Context, coverage: bool = False, verbose: bool = False):
    """Run tests.

    Args:
        coverage: Generate coverage report.
        verbose: Verbose output.
    """
    cmd = "pdm run pytest"
    if coverage:
        cmd += " --cov=src/mdn_lstm --cov-report=term-missing"
    if verbose:
        cmd += " -v"
    ctx.run(cmd)


@task
def lint(ctx: Context, fix: bool = False):
    """Run linters (ruff).

    Args:
        fix: Automatically fix issues.
    """
    cmd = "pdm run ruff check src"
    if fix:
        cmd += " --fix"
    ctx.run(cmd)


@task
def format_code(ctx: Context, check: bool = False):
    """Format code with ruff.

    Args:
        check: Only check, don't modify files.
    """
    cmd = "pdm run ruff format src"
    if check:
        cmd += " --check"
    ctx.run(cmd)


@task
def typecheck(ctx: Context):
    """Run type checking with mypy."""
    ctx.run("pdm run mypy src")


@task
def clean(ctx: Context):
    """Clean build artifacts and caches."""
    patterns = [
        "build/",
        "dist/",
        "*.egg-info/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "__pycache__/",
        "*.pyc",
        ".coverage",
        "htmlcov/",
    ]
    for pattern in patterns:
        ctx.run(f"rm -rf {pattern}", warn=True)


@task
def init_config(ctx: Context, output: str = "config.yaml"):
    """Generate a default configuration file.

    Args:
        output: Output path for the configuration file.
    """
    ctx.run(f"python -m mdn_lstm.cli init-config --output {output}")


@task(pre=[lint, typecheck, test])
def check(ctx: Context):
    """Run all checks (lint, typecheck, test)."""
    print("All checks passed!")
