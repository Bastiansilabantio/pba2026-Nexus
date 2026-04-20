"""
BiLSTM architecture module for sentiment analysis (PyTorch).

Features:
- Clean, reusable BiLSTM classifier implementation
- Optional pretrained embeddings support
- Parameter counting utilities
- Automatic parameter-limit verification (default <= 10,000,000)
- Lightweight self-test runnable from CLI

Author: pba2026-Nexus
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Configuration dataclass
# -----------------------------
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


# -----------------------------
# Model definition
# -----------------------------
class BiLSTMClassifier(nn.Module):
    """
    BiLSTM classifier for text classification.

    Expected input:
        input_ids: Tensor[batch_size, seq_len] (token indices)
        lengths:   Optional Tensor[batch_size] containing true sequence lengths
                   (used for packed sequence to ignore padding in RNN)

    Output:
        logits: Tensor[batch_size, num_classes]
    """

    def __init__(
        self,
        config: BiLSTMConfig,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        # Optional: load pretrained embeddings
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (
                config.vocab_size,
                config.embedding_dim,
            ):
                raise ValueError(
                    "pretrained_embeddings shape mismatch: "
                    f"expected {(config.vocab_size, config.embedding_dim)}, "
                    f"got {tuple(pretrained_embeddings.shape)}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=config.bidirectional,
        )

        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_output_dim, config.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tensor[batch_size, seq_len]
            lengths: Optional Tensor[batch_size], sequence lengths before padding

        Returns:
            logits: Tensor[batch_size, num_classes]
        """
        emb = self.embedding(input_ids)  # [B, T, E]

        if lengths is not None:
            # Pack padded sequence for efficient LSTM processing
            packed = nn.utils.rnn.pack_padded_sequence(
                emb,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(emb)

        # h_n shape: [num_layers * num_directions, B, H]
        if self.config.bidirectional:
            # Take final layer forward + backward hidden states
            forward_last = h_n[-2]  # [B, H]
            backward_last = h_n[-1]  # [B, H]
            features = torch.cat([forward_last, backward_last], dim=1)  # [B, 2H]
        else:
            features = h_n[-1]  # [B, H]

        features = self.dropout(features)
        logits = self.classifier(features)  # [B, C]
        return logits


# -----------------------------
# Parameter utility functions
# -----------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: if True, count only requires_grad=True params

    Returns:
        int: number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_report(model: nn.Module) -> Dict[str, int]:
    """
    Return a parameter report dictionary.
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def verify_parameter_limit(
    model: nn.Module,
    max_parameters: int = 10_000_000,
) -> Tuple[bool, str]:
    """
    Verify parameter count does not exceed the given limit.

    Returns:
        (is_valid, message)
    """
    total = count_parameters(model, trainable_only=False)
    if total <= max_parameters:
        return (
            True,
            f"OK: total parameters {total:,} <= limit {max_parameters:,}",
        )
    return (
        False,
        f"EXCEEDED: total parameters {total:,} > limit {max_parameters:,}",
    )


def create_bilstm_model(
    config: BiLSTMConfig,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    freeze_embeddings: bool = False,
    enforce_limit: bool = True,
) -> BiLSTMClassifier:
    """
    Factory helper to build BiLSTM and (optionally) enforce parameter constraint.
    """
    model = BiLSTMClassifier(
        config=config,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
    )

    if enforce_limit:
        ok, msg = verify_parameter_limit(model, config.max_parameters)
        if not ok:
            raise ValueError(msg)

    return model


# -----------------------------
# Lightweight self-test CLI
# -----------------------------
if __name__ == "__main__":
    # Example config intentionally under 10M params
    cfg = BiLSTMConfig(
        vocab_size=20_000,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
        pad_idx=0,
        bidirectional=True,
        max_parameters=10_000_000,
    )

    model = create_bilstm_model(cfg, enforce_limit=True)

    report = parameter_report(model)
    ok, msg = verify_parameter_limit(model, cfg.max_parameters)

    print("BiLSTM model created.")
    print(f"Total params      : {report['total']:,}")
    print(f"Trainable params  : {report['trainable']:,}")
    print(f"Non-trainable     : {report['non_trainable']:,}")
    print(msg)

    # Dummy forward pass
    batch_size, seq_len = 4, 32
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.tensor([32, 28, 19, 12])

    logits = model(dummy_input, dummy_lengths)
    print(f"Forward OK. Logits shape: {tuple(logits.shape)}")
