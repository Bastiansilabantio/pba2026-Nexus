---
title: Nexus Sentiment Analyzer
emoji: 🐦
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Nexus Sentiment Analyzer (Hugging Face Space)

Aplikasi ini adalah demo analisis sentimen teks berbahasa Inggris menggunakan model Machine Learning berbasis **TF-IDF + Logistic Regression**.

## 📦 Isi Folder Space

- `app.py` — aplikasi Streamlit
- `best_sentiment_model.pkl` — model klasifikasi sentimen terlatih
- `tfidf_vectorizer.pkl` — vectorizer TF-IDF
- `requirements.txt` — dependency Python untuk runtime
- `README.md` — metadata Space + dokumentasi singkat

## 🚀 Catatan Deploy (Docker SDK)

Space ini memakai **Docker SDK**, jadi jalankan aplikasi melalui `Dockerfile` yang sesuai.
Pastikan image/container menginstall dependency dari `requirements.txt` dan mengeksekusi aplikasi Streamlit.

Contoh command run di dalam container:
`streamlit run app.py --server.port 7860 --server.address 0.0.0.0`

## ✅ Catatan Penting

- `app.py` membaca file model dari direktori yang sama.
- Jika nama file model/vectorizer diubah, sesuaikan path di `app.py`.
- Jangan simpan token/secret di dalam kode. Gunakan pengaturan secret/environment variable di Hugging Face Space.