"""
Evaluate trained BiLSTM sentiment model on test split.

This evaluator is aligned to the standardized artifact structure in:
- models/dl/bilstm_sentiment.pt
- models/dl/vocab.json
- models/dl/config.json
- models/dl/train_config.json (optional fallback)

Expected dataset columns:
- clean_text
- sentiment

Usage:
    python scripts/eval_bilstm.py

Optional:
    python scripts/eval_bilstm.py --data-path data/cleaned_sample.csv --model-dir models/dl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =====================================
# Model Definition
# =====================================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        pad_idx: int = 0,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(emb)

        if self.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            features = torch.cat([forward_last, backward_last], dim=1)
        else:
            features = h_n[-1]

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


# =====================================
# Utility
# =====================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str) -> List[str]:
    return str(text).strip().split()


def encode_text(
    text: str,
    vocab: Dict[str, int],
    max_len: int,
    unk_idx: int,
    pad_idx: int,
) -> Tuple[List[int], int]:
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(tok, unk_idx) for tok in tokens][:max_len]
    length = len(token_ids)

    if length < max_len:
        token_ids += [pad_idx] * (max_len - length)

    return token_ids, max(length, 1)


def build_eval_tensors(
    df: pd.DataFrame,
    vocab: Dict[str, int],
    text_col: str,
    label_col: str,
    max_len: int,
    pad_idx: int,
    unk_idx: int,
):
    all_ids: List[List[int]] = []
    all_lengths: List[int] = []
    labels = df[label_col].astype(int).tolist()

    for text in df[text_col].tolist():
        ids, length = encode_text(
            text=text,
            vocab=vocab,
            max_len=max_len,
            unk_idx=unk_idx,
            pad_idx=pad_idx,
        )
        all_ids.append(ids)
        all_lengths.append(length)

    x = torch.tensor(all_ids, dtype=torch.long)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, lengths, y


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifacts(model_dir: Path) -> dict:
    """
    Resolve standardized artifacts from models/dl with robust fallback:

    Priority:
    1) model: bilstm_sentiment.pt
       vocab: vocab.json
       config: config.json
    2) fallback legacy names:
       state: bilstm_state_dict.pt
       vocab: vocab.json
       config: train_config.json or config.json
    """
    std_model = model_dir / "bilstm_sentiment.pt"
    std_vocab = model_dir / "vocab.json"
    std_config = model_dir / "config.json"

    legacy_state = model_dir / "bilstm_state_dict.pt"
    legacy_config = model_dir / "train_config.json"

    artifacts = {
        "model_file": None,
        "vocab_file": None,
        "config_file": None,
        "mode": None,  # "checkpoint" or "state_dict"
    }

    # Standard mode
    if std_model.exists() and std_vocab.exists() and std_config.exists():
        artifacts["model_file"] = std_model
        artifacts["vocab_file"] = std_vocab
        artifacts["config_file"] = std_config
        artifacts["mode"] = "checkpoint"
        return artifacts

    # Legacy fallback mode
    if (
        legacy_state.exists()
        and std_vocab.exists()
        and (legacy_config.exists() or std_config.exists())
    ):
        artifacts["model_file"] = legacy_state
        artifacts["vocab_file"] = std_vocab
        artifacts["config_file"] = (
            legacy_config if legacy_config.exists() else std_config
        )
        artifacts["mode"] = "state_dict"
        return artifacts

    missing = []
    if not std_model.exists() and not legacy_state.exists():
        missing.append("bilstm_sentiment.pt (or bilstm_state_dict.pt)")
    if not std_vocab.exists():
        missing.append("vocab.json")
    if not std_config.exists() and not legacy_config.exists():
        missing.append("config.json (or train_config.json)")

    raise FileNotFoundError(
        "Artifact model DL tidak lengkap di folder models/dl. "
        f"Missing: {', '.join(missing)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BiLSTM sentiment model on test split."
    )
    parser.add_argument("--data-path", type=str, default="data/cleaned_sample.csv")
    parser.add_argument("--model-dir", type=str, default="models/dl")
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="sentiment")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-json", type=str, default="models/dl/eval_metrics.json"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Harus sama dengan val-size saat training agar split konsisten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    output_json = Path(args.output_json)

    if not data_path.exists():
        raise FileNotFoundError(f"Data tidak ditemukan: {data_path}")

    artifacts = resolve_artifacts(model_dir)

    print(f"[INFO] Load dataset: {data_path}")
    df = pd.read_csv(data_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"Kolom tidak ditemukan. Butuh '{args.text_col}' dan '{args.label_col}'. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)

    print(f"[INFO] Total data: {len(df)}")
    print(f"[INFO] Distribusi label: {Counter(df[args.label_col].tolist())}")

    # Reconstruct split strategy exactly like training:
    # 1) split train_temp/test with test_size
    # 2) split train/val from train_temp with val_rel_size
    train_temp_df, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.label_col],
    )

    val_rel_size = args.val_size / (1.0 - args.test_size)
    if val_rel_size <= 0 or val_rel_size >= 1:
        raise ValueError(
            f"Nilai val_rel_size tidak valid: {val_rel_size}. "
            "Periksa kombinasi test_size pada evaluasi/training."
        )

    _, _ = train_test_split(
        train_temp_df,
        test_size=val_rel_size,
        random_state=args.random_state,
        stratify=train_temp_df[args.label_col],
    )

    print(f"[INFO] Ukuran test set: {len(df_test)}")

    print("[INFO] Load artifacts...")
    vocab = load_json(artifacts["vocab_file"])
    cfg = load_json(artifacts["config_file"])

    pad_token = cfg.get("pad_token", "<PAD>")
    unk_token = cfg.get("unk_token", "<UNK>")
    pad_idx = int(cfg.get("pad_idx", vocab.get(pad_token, 0)))
    unk_idx = int(cfg.get("unk_idx", vocab.get(unk_token, 1)))

    max_len = int(cfg.get("max_len", 50))
    num_classes = int(cfg.get("num_classes", 2))
    vocab_size = int(cfg.get("vocab_size", len(vocab)))
    embedding_dim = int(cfg.get("embedding_dim", 128))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.3))
    bidirectional = bool(cfg.get("bidirectional", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        pad_idx=pad_idx,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    # Load weights depending on artifact mode
    if artifacts["mode"] == "checkpoint":
        checkpoint = torch.load(artifacts["model_file"], map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # direct state dict
            model.load_state_dict(checkpoint)
    else:
        # legacy direct state_dict file
        state_dict = torch.load(artifacts["model_file"], map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    x_test, len_test, y_test = build_eval_tensors(
        df=df_test,
        vocab=vocab,
        text_col=args.text_col,
        label_col=args.label_col,
        max_len=max_len,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
    )

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        n = x_test.size(0)
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            xb = x_test[start:end].to(device)
            lb = len_test[start:end].to(device)
            yb = y_test[start:end].cpu().numpy()

            logits = model(xb, lb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(yb.tolist())
            y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        zero_division=0,
        output_dict=True,
    )

    results = {
        "num_test_samples": len(y_true),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "artifacts_used": {
            "model_file": str(artifacts["model_file"]),
            "vocab_file": str(artifacts["vocab_file"]),
            "config_file": str(artifacts["config_file"]),
            "mode": artifacts["mode"],
        },
        "config_used": {
            "data_path": str(data_path),
            "model_dir": str(model_dir),
            "text_col": args.text_col,
            "label_col": args.label_col,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "batch_size": args.batch_size,
            "val_size": args.val_size,
            "val_rel_size": val_rel_size,
            "max_len": max_len,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Evaluation Results (BiLSTM) ===")
    print(f"Test samples : {len(y_true)}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Confusion M. : {cm}")
    print(f"\n[INFO] Metrics saved to: {output_json}")


if __name__ == "__main__":
    main()
