import streamlit as st
import pickle
import re

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Sentimen Tweet",
    page_icon="🐦",
    layout="centered"
)

# ── Load Model & Vectorizer ───────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as g:
        vectorizer = pickle.load(g)
    return model, vectorizer

model, vectorizer = load_model()

# ── Text Cleaning ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        .positive {
            background: linear-gradient(135deg, #d4edda, #a8e6cf);
            color: #155724;
            border: 1px solid #28a745;
        }
        .negative {
            background: linear-gradient(135deg, #f8d7da, #f5a0a8);
            color: #721c24;
            border: 1px solid #dc3545;
        }
        .info-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 0.8rem 1.2rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #374151;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🐦 Deteksi Sentimen Tweet</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan teks bahasa Inggris untuk mendeteksi apakah sentimennya Positif atau Negatif</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">Model: <strong>Logistic Regression</strong> | Dataset: Sentiment140 (10.000 sampel) | Akurasi: <strong>73.5%</strong></div>', unsafe_allow_html=True)

# Input
user_input = st.text_area(
    "Masukkan teks tweet:",
    height=120,
    placeholder="Contoh: I love this beautiful sunny day! or I'm so sad and upset today..."
)

# Contoh cepat
st.markdown("**Coba contoh:**")
col1, col2, col3 = st.columns(3)
examples = {
    "😊 Positif": "I love this product so much, it made my day!",
    "😢 Negatif": "This is the worst experience I've ever had.",
    "Netral": "Just had lunch and now going back to work."
}
for (label, example), col in zip(examples.items(), [col1, col2, col3]):
    if col.button(label, use_container_width=True):
        user_input = example
        st.session_state["text_input"] = example

# Predict
if st.button("🔍 Deteksi Sentimen", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.markdown(
                '<div class="result-card positive">😊 Sentimen: POSITIF</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card negative">😢 Sentimen: NEGATIF</div>',
                unsafe_allow_html=True
            )

        if prob is not None:
            st.markdown(f"\n**Confidence** — Negatif: `{prob[0]*100:.1f}%` | Positif: `{prob[1]*100:.1f}%`")

st.divider()
st.markdown(
    "<div style='text-align:center; font-size:0.8rem; color:#9ca3af;'>NLP Tugas Besar PBA — Informatika ITERA 2026</div>",
    unsafe_allow_html=True
)
