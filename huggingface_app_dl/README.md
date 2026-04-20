---
title: Nexus Sentiment Analyzer - BiLSTM DL
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Nexus Sentiment Analyzer (Deep Learning - BiLSTM)

Demo ini menampilkan analisis sentimen teks bahasa Inggris menggunakan model **BiLSTM (PyTorch)**.

## 📦 Isi Folder Space

- `app.py` — aplikasi demo Streamlit
- `Dockerfile` — runtime container untuk Hugging Face Space (Docker SDK)
- `requirements.txt` — dependency Python
- `bilstm_state_dict.pt` — bobot model BiLSTM (state_dict)
- `vocab.json` — vocabulary token ke index
- `config.json` — konfigurasi model & preprocessing

## 🚀 Deploy Notes

Space ini menggunakan **Docker SDK**.  
Aplikasi dijalankan dengan Streamlit di port `7860`.

Contoh command runtime di container:
`streamlit run app.py --server.port 7860 --server.address 0.0.0.0`

## ✅ Catatan Penting

- Pastikan file artifact model tersedia di root Space:
  - `bilstm_state_dict.pt`
  - `vocab.json`
  - `config.json`
- Jangan menyimpan token/secret di source code. Gunakan pengaturan secret dari Hugging Face Space.
- Jika mengubah nama file artifact, sesuaikan juga path pembacaan pada `app.py`.