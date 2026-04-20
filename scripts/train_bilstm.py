"""
Train BiLSTM sentiment model end-to-end (PyTorch).

What this script does:
1) Load cleaned dataset CSV with columns: `clean_text`, `sentiment`
2) Split into train/validation/test
3) Build vocabulary from training data
4) Convert text to padded token-id sequences
5) Train BiLSTM model
6) Save:
   - best model weights (.pt)
   - training artifacts (config, vocab)
   - loss curve image
   - history CSV

Default paths are aligned with current repository structure.

Usage:
    python scripts/train_bilstm.py

Optional:
    python scripts/train_bilstm.py --epochs 10 --batch-size 64 --max-len 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Tokenization utils
# -------------------------------------------------
def basic_tokenize(text: str) -> List[str]:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    return text.split(" ")


def build_vocab(
    texts: List[str],
    min_freq: int = 2,
    max_vocab_size: int = 20_000,
    specials: Tuple[str, str] = ("<PAD>", "<UNK>"),
) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    # Reserve indices for special tokens
    stoi = {specials[0]: 0, specials[1]: 1}

    # Most common tokens
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in stoi:
            continue
        if len(stoi) >= max_vocab_size:
            break
        stoi[token] = len(stoi)

    return stoi


def encode_text(text: str, stoi: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    tokens = basic_tokenize(text)
    unk_id = stoi["<UNK>"]
    ids = [stoi.get(tok, unk_id) for tok in tokens][:max_len]
    length = len(ids)

    if length < max_len:
        ids += [stoi["<PAD>"]] * (max_len - length)

    return ids, length


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(
        self, texts: List[str], labels: List[int], stoi: Dict[str, int], max_len: int
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = int(self.labels[idx])

        input_ids, length = encode_text(text, self.stoi, self.max_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# -------------------------------------------------
# Model config + architecture
# -------------------------------------------------
@dataclass
class BiLSTMConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    num_classes: int = 2
    pad_idx: int = 0
    bidirectional: bool = True
    max_parameters: int = 10_000_000


class BiLSTMClassifier(nn.Module):
    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=config.bidirectional,
        )

        lstm_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_out_dim, config.num_classes)

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, T, E]

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(emb)

        if self.config.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            feat = torch.cat([forward_last, backward_last], dim=1)
        else:
            feat = h_n[-1]

        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------
# Train / eval loops
# -------------------------------------------------
def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, lengths, labels in tqdm(loader, desc="Train", leave=False):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, lengths, labels in tqdm(loader, desc="Eval ", leave=False):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# -------------------------------------------------
# Helpers: save artifacts
# -------------------------------------------------
def save_loss_plot(history: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiLSTM Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------------------------------
# Args
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM sentiment model.")
    parser.add_argument("--data-path", type=str, default="data/cleaned_sample.csv")
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="sentiment")

    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)

    parser.add_argument("--max-vocab-size", type=int, default=20_000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=50)

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-dir", type=str, default="models/dl")
    parser.add_argument("--assets-dir", type=str, default="assets/dl")
    parser.add_argument("--run-name", type=str, default="bilstm_sentiment_v1")

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    model_dir = Path(args.model_dir)
    assets_dir = Path(args.assets_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"Columns not found. Required: {args.text_col}, {args.label_col}. "
            f"Got: {list(df.columns)}"
        )

    df = df[[args.text_col, args.label_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)

    # Split: first test, then val from train
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[args.label_col],
    )

    # val fraction relative to current train_df
    val_rel_size = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_rel_size,
        random_state=args.seed,
        stratify=train_df[args.label_col],
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")
    print(f"Test size : {len(test_df)}")

    # Build vocab from training set only
    stoi = build_vocab(
        texts=train_df[args.text_col].tolist(),
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Vocab size (actual): {len(stoi)}")

    # Datasets + loaders
    train_ds = SentimentDataset(
        train_df[args.text_col].tolist(),
        train_df[args.label_col].tolist(),
        stoi,
        args.max_len,
    )
    val_ds = SentimentDataset(
        val_df[args.text_col].tolist(),
        val_df[args.label_col].tolist(),
        stoi,
        args.max_len,
    )
    test_ds = SentimentDataset(
        test_df[args.text_col].tolist(),
        test_df[args.label_col].tolist(),
        stoi,
        args.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model config
    cfg = BiLSTMConfig(
        vocab_size=len(stoi),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        pad_idx=stoi["<PAD>"],
        bidirectional=True,
        max_parameters=10_000_000,
    )

    model = BiLSTMClassifier(cfg)
    trainable_params = count_trainable_params(model)
    print(f"Trainable params: {trainable_params:,}")

    if trainable_params > cfg.max_parameters:
        raise ValueError(
            f"Parameter limit exceeded: {trainable_params:,} > {cfg.max_parameters:,}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Train loop
    best_val_loss = math.inf
    best_state_dict = None
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch_train(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = run_epoch_eval(model, val_loader, criterion, device)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    # Restore best and evaluate on test
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = run_epoch_eval(model, test_loader, criterion, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save model + artifacts
    run_name = args.run_name
    # Keep both standardized and run-specific artifact names for compatibility.
    # Standard names are used by evaluator/deploy scripts.
    model_path = model_dir / "bilstm_state_dict.pt"
    checkpoint_path = model_dir / "bilstm_sentiment.pt"
    vocab_path = model_dir / "vocab.json"
    config_path = model_dir / "train_config.json"
    deploy_config_path = model_dir / "config.json"
    metrics_path = model_dir / "train_metrics.json"

    run_model_path = model_dir / f"{run_name}.pt"
    run_vocab_path = model_dir / f"{run_name}_vocab.json"
    run_config_path = model_dir / f"{run_name}_config.json"
    run_metrics_path = model_dir / f"{run_name}_metrics.json"

    history_csv_path = assets_dir / f"{run_name}_history.csv"
    loss_plot_path = assets_dir / f"{run_name}_loss_curve.png"

    # 1) Legacy/state-dict only artifact
    torch.save(model.state_dict(), model_path)

    # 2) Full checkpoint artifact (recommended for demo/deploy)
    checkpoint_obj = {
        "state_dict": model.state_dict(),
        "model_config": asdict(cfg),
        "stoi": stoi,
        "max_len": args.max_len,
        "label_map": {"0": "negative", "1": "positive"},
        "run_name": run_name,
    }
    torch.save(checkpoint_obj, checkpoint_path)
    torch.save(checkpoint_obj, run_model_path)

    save_json(stoi, vocab_path)
    save_json(stoi, run_vocab_path)

    train_config = asdict(cfg)
    train_config.update(
        {
            "max_len": args.max_len,
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "pad_idx": stoi["<PAD>"],
            "unk_idx": stoi["<UNK>"],
            "vocab_size": len(stoi),
            "num_classes": 2,
            "label_map": {"0": "negative", "1": "positive"},
            "run_name": run_name,
        }
    )
    save_json(train_config, config_path)
    save_json(train_config, deploy_config_path)
    save_json(train_config, run_config_path)

    metrics_obj = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "trainable_params": trainable_params,
        "max_allowed_params": cfg.max_parameters,
        "parameter_limit_ok": trainable_params <= cfg.max_parameters,
    }
    save_json(metrics_obj, metrics_path)
    save_json(metrics_obj, run_metrics_path)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_csv_path, index=False)
    save_loss_plot(history_df, loss_plot_path)

    print("\nTraining complete. Artifacts saved:")
    print(f"- State Dict (standard) : {model_path}")
    print(f"- Checkpoint (standard) : {checkpoint_path}")
    print(f"- Checkpoint (run)      : {run_model_path}")
    print(f"- Vocab (standard)      : {vocab_path}")
    print(f"- Vocab (run)           : {run_vocab_path}")
    print(f"- Config (standard)     : {config_path}")
    print(f"- Config deploy         : {deploy_config_path}")
    print(f"- Config (run)          : {run_config_path}")
    print(f"- Metrics (standard)    : {metrics_path}")
    print(f"- Metrics (run)         : {run_metrics_path}")
    print(f"- History CSV           : {history_csv_path}")
    print(f"- Loss Plot             : {loss_plot_path}")


if __name__ == "__main__":
    main()
