import re
import pickle
from pathlib import Path

import streamlit as st


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Deteksi Sentimen Tweet",
    page_icon="🐦",
    layout="centered",
)


# =========================
# Helper: resolve artifact paths
# =========================
BASE_DIR = Path(__file__).resolve().parent


def find_existing_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


MODEL_PATH = find_existing_path(
    [
        BASE_DIR / "best_sentiment_model.pkl",
        BASE_DIR / "models" / "best_sentiment_model.pkl",
    ]
)

VECTORIZER_PATH = find_existing_path(
    [
        BASE_DIR / "tfidf_vectorizer.pkl",
        BASE_DIR / "models" / "tfidf_vectorizer.pkl",
    ]
)


# =========================
# Load model and vectorizer
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: Path, vectorizer_path: Path):
    with model_path.open("rb") as f:
        model = pickle.load(f)
    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# UI styling
# =========================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main-title {
            text-align: center;
            font-size: 2.0rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #111827;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1.2rem;
        }

        .info-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 0.85rem 1rem;
            border-radius: 8px;
            font-size: 0.88rem;
            color: #1f2937;
            margin-bottom: 1rem;
        }

        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .positive {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #22c55e;
        }

        .negative {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Header
# =========================
st.markdown('<div class="main-title">🐦 Deteksi Sentimen Tweet</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Masukkan teks bahasa Inggris untuk memprediksi sentimen: Positif atau Negatif</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-box"><strong>Model:</strong> Logistic Regression &nbsp;|&nbsp; <strong>Fitur:</strong> TF-IDF</div>',
    unsafe_allow_html=True,
)


# =========================
# Validate artifacts
# =========================
if MODEL_PATH is None or VECTORIZER_PATH is None:
    st.error(
        "File model/vectorizer tidak ditemukan.\n\n"
        "Pastikan file berikut tersedia di folder yang sama dengan app.py:\n"
        "- best_sentiment_model.pkl\n"
        "- tfidf_vectorizer.pkl"
    )
    st.stop()


# =========================
# Load artifacts safely
# =========================
try:
    model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)
except Exception as e:
    st.error(f"Gagal memuat model/vectorizer: {e}")
    st.stop()


# =========================
# Session state init
# =========================
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""


# =========================
# Input
# =========================
st.text_area(
    "Masukkan teks tweet:",
    key="tweet_input",
    height=130,
    placeholder="Contoh: I love this product so much!",
)

st.markdown("**Contoh cepat:**")
col1, col2, col3 = st.columns(3)

if col1.button("😊 Positif", use_container_width=True):
    st.session_state["tweet_input"] = "I absolutely love this, it made my day!"
    st.rerun()

if col2.button("😢 Negatif", use_container_width=True):
    st.session_state["tweet_input"] = "This is the worst service I have ever used."
    st.rerun()

if col3.button("😐 Netral", use_container_width=True):
    st.session_state["tweet_input"] = "I had lunch and now I am back to work."
    st.rerun()


# =========================
# Prediction
# =========================
if st.button("🔍 Deteksi Sentimen", type="primary", use_container_width=True):
    raw_text = st.session_state.get("tweet_input", "").strip()

    if not raw_text:
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(raw_text)

        if not cleaned:
            st.warning("Teks menjadi kosong setelah preprocessing.")
        else:
            try:
                features = vectorizer.transform([cleaned])
                pred = model.predict(features)[0]

                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]

                if int(pred) == 1:
                    st.markdown(
                        '<div class="result-card positive">😊 Sentimen: POSITIF</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-card negative">😢 Sentimen: NEGATIF</div>',
                        unsafe_allow_html=True,
                    )

                if probs is not None and len(probs) >= 2:
                    st.markdown(
                        f"**Confidence** — Negatif: `{probs[0]*100:.1f}%` | Positif: `{probs[1]*100:.1f}%`"
                    )

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")


st.divider()
st.caption("NLP Project — Streamlit deployment ready for Hugging Face Spaces")
