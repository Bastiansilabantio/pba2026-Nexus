"""
Quick parameter checker for a BiLSTM sentiment model.

Purpose:
- Verify total trainable parameters are <= 10,000,000 as required.

Usage examples:
1) Default config:
   python scripts/check_bilstm_params.py

2) Custom config:
   python scripts/check_bilstm_params.py --vocab-size 20000 --embed-dim 128 --hidden-dim 128 --num-layers 2 --num-classes 2

3) With tied embeddings off/on:
   python scripts/check_bilstm_params.py --tie-embeddings

Notes:
- This script only checks architecture parameter count, not training performance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

MAX_PARAMS = 10_000_000


@dataclass
class ModelConfig:
    vocab_size: int = 20_000
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.3
    pad_idx: int = 0
    tie_embeddings: bool = False


class BiLSTMSentiment(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.embed_dim,
            padding_idx=cfg.pad_idx,
        )

        # Dropout in nn.LSTM only active if num_layers > 1
        lstm_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(cfg.dropout)

        # BiLSTM output size = hidden_dim * 2
        self.classifier = nn.Linear(cfg.hidden_dim * 2, cfg.num_classes)

        # Optional: tie classifier weight with embedding when dimensions match.
        # This is uncommon for classification, but provided as an option.
        if cfg.tie_embeddings:
            if cfg.num_classes != cfg.vocab_size:
                raise ValueError(
                    "Cannot tie embeddings: num_classes must equal vocab_size "
                    f"(got num_classes={cfg.num_classes}, vocab_size={cfg.vocab_size})."
                )
            if cfg.hidden_dim * 2 != cfg.embed_dim:
                raise ValueError(
                    "Cannot tie embeddings: hidden_dim*2 must equal embed_dim "
                    f"(got hidden_dim*2={cfg.hidden_dim * 2}, embed_dim={cfg.embed_dim})."
                )
            self.classifier.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len] (token ids)
        """
        emb = self.embedding(x)  # [B, T, E]
        out, _ = self.lstm(emb)  # [B, T, 2H]
        pooled = out.mean(dim=1)  # mean pooling over time [B, 2H]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, C]
        return logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_int(n: int) -> str:
    return f"{n:,}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly verify BiLSTM trainable parameter count <= 10 million."
    )
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pad-idx", type=int, default=0)
    parser.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="Try tying classifier weight with embedding weight (rare for classification).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        pad_idx=args.pad_idx,
        tie_embeddings=args.tie_embeddings,
    )

    model = BiLSTMSentiment(cfg)
    total_params = count_trainable_params(model)
    is_valid = total_params <= MAX_PARAMS

    if args.json:
        output = {
            "config": asdict(cfg),
            "trainable_params": total_params,
            "max_allowed_params": MAX_PARAMS,
            "is_valid": is_valid,
            "margin": MAX_PARAMS - total_params,
        }
        print(json.dumps(output, indent=2))
        return

    print("=== BiLSTM Parameter Check ===")
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"- {k}: {v}")

    print("\nResult:")
    print(f"- Trainable params : {format_int(total_params)}")
    print(f"- Max allowed      : {format_int(MAX_PARAMS)}")

    if is_valid:
        print(
            f"- Status           : PASS ✅ (margin: {format_int(MAX_PARAMS - total_params)})"
        )
    else:
        print(
            f"- Status           : FAIL ❌ (exceed: {format_int(total_params - MAX_PARAMS)})"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
