import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import torch.nn as nn

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Deteksi Sentimen Tweet (BiLSTM DL)",
    page_icon="🧠",
    layout="centered",
)

# =========================
# Paths (HF Space root)
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Standard artifact names copied to Space root
MODEL_PATH = BASE_DIR / "bilstm_state_dict.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"
CONFIG_PATH = BASE_DIR / "config.json"


# =========================
# Model definition
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_idx: int = 0,
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
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
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
        return self.classifier(features)


# =========================
# Text processing
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+|#", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def encode_tokens(
    tokens: List[str],
    vocab: Dict[str, int],
    unk_idx: int,
    pad_idx: int,
    max_len: int,
) -> Tuple[List[int], int]:
    ids = [vocab.get(tok, unk_idx) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [pad_idx] * (max_len - length)
    return ids, max(length, 1)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Artifact loader
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file tidak ditemukan: {MODEL_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocab file tidak ditemukan: {VOCAB_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {CONFIG_PATH}")

    vocab = load_json(VOCAB_PATH)
    cfg = load_json(CONFIG_PATH)

    pad_token = cfg.get("pad_token", "<PAD>")
    unk_token = cfg.get("unk_token", "<UNK>")

    pad_idx = int(cfg.get("pad_idx", vocab.get(pad_token, 0)))
    unk_idx = int(cfg.get("unk_idx", vocab.get(unk_token, 1)))
    max_len = int(cfg.get("max_len", 50))

    model = BiLSTMClassifier(
        vocab_size=int(cfg.get("vocab_size", len(vocab))),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
        num_classes=int(cfg.get("num_classes", 2)),
        pad_idx=pad_idx,
        bidirectional=bool(cfg.get("bidirectional", True)),
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    label_map = cfg.get("label_map", {"0": "negative", "1": "positive"})
    if not isinstance(label_map, dict):
        label_map = {"0": "negative", "1": "positive"}
    label_map = {str(k): str(v) for k, v in label_map.items()}

    return {
        "model": model,
        "vocab": vocab,
        "cfg": cfg,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "max_len": max_len,
        "label_map": label_map,
    }


# =========================
# Inference
# =========================
def predict_sentiment(text: str, artifacts: dict) -> dict:
    model = artifacts["model"]
    vocab = artifacts["vocab"]
    unk_idx = artifacts["unk_idx"]
    pad_idx = artifacts["pad_idx"]
    max_len = artifacts["max_len"]
    label_map = artifacts["label_map"]

    cleaned = clean_text(text)
    tokens = cleaned.split()

    ids, length = encode_tokens(
        tokens=tokens,
        vocab=vocab,
        unk_idx=unk_idx,
        pad_idx=pad_idx,
        max_len=max_len,
    )

    input_ids = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    return {
        "pred_idx": pred_idx,
        "label": label_map.get(str(pred_idx), str(pred_idx)),
        "probs": probs.tolist(),
        "cleaned_text": cleaned,
        "token_count": len(tokens),
        "seq_len_used": length,
    }


# =========================
# UI styling
# =========================
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        background: #f9fafb;
    }
    .pos {
        color: #166534;
        font-weight: 700;
    }
    .neg {
        color: #991b1b;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">🧠 Deteksi Sentimen Tweet (BiLSTM)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Deploy Deep Learning PyTorch di Hugging Face Spaces</div>',
    unsafe_allow_html=True,
)

# Load artifacts
try:
    artifacts = load_artifacts()
    st.success("Artifact model DL berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat artifact: {e}")
    st.stop()

with st.expander("Info Model", expanded=False):
    st.json(
        {
            "model_path": str(MODEL_PATH),
            "vocab_path": str(VOCAB_PATH),
            "config_path": str(CONFIG_PATH),
            "vocab_size": artifacts["cfg"].get("vocab_size"),
            "embedding_dim": artifacts["cfg"].get("embedding_dim"),
            "hidden_dim": artifacts["cfg"].get("hidden_dim"),
            "num_layers": artifacts["cfg"].get("num_layers"),
            "bidirectional": artifacts["cfg"].get("bidirectional"),
            "max_len": artifacts["max_len"],
        }
    )

if "dl_input_text" not in st.session_state:
    st.session_state["dl_input_text"] = ""

st.text_area(
    "Masukkan teks tweet (Bahasa Inggris):",
    key="dl_input_text",
    height=140,
    placeholder="Contoh: I really love this movie! It was amazing.",
)

st.markdown("**Contoh cepat:**")
c1, c2, c3 = st.columns(3)

if c1.button("😊 Positif", use_container_width=True):
    st.session_state["dl_input_text"] = (
        "I absolutely love this product, it works perfectly."
    )
    st.rerun()

if c2.button("😢 Negatif", use_container_width=True):
    st.session_state["dl_input_text"] = (
        "This is the worst experience ever, very disappointing."
    )
    st.rerun()

if c3.button("😐 Netral", use_container_width=True):
    st.session_state["dl_input_text"] = "I had lunch and then returned to the office."
    st.rerun()

if st.button("🔍 Prediksi Sentimen (DL)", type="primary", use_container_width=True):
    text = st.session_state.get("dl_input_text", "").strip()
    if not text:
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        result = predict_sentiment(text, artifacts=artifacts)
        label_upper = result["label"].upper()

        if "POS" in label_upper:
            st.markdown(
                '<div class="card">Hasil: <span class="pos">😊 POSITIF</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="card">Hasil: <span class="neg">😢 NEGATIF</span></div>',
                unsafe_allow_html=True,
            )

        probs = result["probs"]
        if len(probs) >= 2:
            st.write(
                {
                    "confidence_negatif": f"{probs[0] * 100:.2f}%",
                    "confidence_positif": f"{probs[1] * 100:.2f}%",
                }
            )

        with st.expander("Detail preprocessing", expanded=False):
            st.write(
                {
                    "cleaned_text": result["cleaned_text"],
                    "token_count": result["token_count"],
                    "seq_len_used": result["seq_len_used"],
                    "max_len": artifacts["max_len"],
                }
            )

st.divider()
st.caption("NLP Project — BiLSTM Deep Learning Demo on Hugging Face Spaces")
