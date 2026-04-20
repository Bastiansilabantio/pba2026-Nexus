"""
Plot Deep Learning training metrics (loss & accuracy) from history CSV.

Target file:
- assets/dl/bilstm_sentiment_v1_history.csv

Generated outputs:
- assets/dl/bilstm_sentiment_v1_loss_curve.png
- assets/dl/bilstm_sentiment_v1_accuracy_curve.png
- assets/dl/bilstm_sentiment_v1_metrics_combined.png

Usage:
    python scripts/plot_dl_metrics.py

Optional:
    python scripts/plot_dl_metrics.py --history-path assets/dl/bilstm_sentiment_v1_history.csv --output-dir assets/dl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DL metrics: training/validation loss and accuracy."
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="assets/dl/bilstm_sentiment_v1_history.csv",
        help="Path ke file history CSV hasil training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/dl",
        help="Folder output untuk menyimpan gambar plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI output gambar.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada CSV: {sorted(missing)}. "
            f"Kolom tersedia: {list(df.columns)}"
        )


def save_loss_plot(df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    plt.title("BiLSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_accuracy_plot(df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
    plt.title("BiLSTM Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_combined_plot(df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Accuracy subplot
    axes[1].plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("BiLSTM Training Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    history_path = Path(args.history_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History CSV tidak ditemukan: {history_path}\n"
            "Pastikan file hasil training tersedia."
        )

    df = pd.read_csv(history_path)
    validate_columns(df)

    run_name = history_path.stem.replace("_history", "")
    loss_path = output_dir / f"{run_name}_loss_curve.png"
    acc_path = output_dir / f"{run_name}_accuracy_curve.png"
    combined_path = output_dir / f"{run_name}_metrics_combined.png"

    save_loss_plot(df, loss_path, dpi=args.dpi)
    save_accuracy_plot(df, acc_path, dpi=args.dpi)
    save_combined_plot(df, combined_path, dpi=args.dpi)

    print("✅ Plot berhasil dibuat:")
    print(f"- Loss curve     : {loss_path}")
    print(f"- Accuracy curve : {acc_path}")
    print(f"- Combined plot  : {combined_path}")


if __name__ == "__main__":
    main()
