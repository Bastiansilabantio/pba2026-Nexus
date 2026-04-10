# pba2026-Nexus

Proyek analisis sentimen tweet berbasis **Machine Learning (TF-IDF + Logistic Regression)** dengan antarmuka **Streamlit**.  
Repository ini berisi alur mulai dari preprocessing data, hasil EDA, model terlatih, hingga aplikasi inferensi.

## 👥 Anggota Kelompok
- Vita Anggraini (122450046)
- Cintya Bella (122450066)
- Bastian (122450130)

## 🔗 Tautan Penting
- Dataset: [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Demo App (Hugging Face/Streamlit): [tubes-nlp-bcv.streamlit.app](https://tubes-nlp-bcv.streamlit.app/)

## 📁 Struktur Repository
```text
pba2026-Nexus/
├── app/
│   └── app.py
├── assets/
│   ├── sentiment_distribution.png
│   └── text_length_distribution.png
├── data/
│   ├── cleaned_sample.csv
│   └── hasil_scrapping_komentar_youtube.xlsx
├── huggingface_app/
│   ├── app.py
│   ├── best_sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── requirements.txt
├── models/
│   ├── best_sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── EDA_dan_Preprocessing.ipynb
├── scripts/
│   └── eda_preprocessing.py
├── requirements.txt
└── README.md
```

## 🚀 Menjalankan Aplikasi Secara Lokal

### 1) Clone repository
```bash
git clone <url-repository-anda>
cd pba2026-Nexus
```

### 2) Buat dan aktifkan virtual environment (opsional tapi direkomendasikan)
```bash
python -m venv .venv
```

- Windows:
```bash
.venv\Scripts\activate
```

- Linux/macOS:
```bash
source .venv/bin/activate
```

### 3) Install dependency
```bash
pip install -r requirements.txt
```

### 4) Jalankan Streamlit app
```bash
streamlit run app/app.py
```

## 🧠 Model
- **Algoritma**: Logistic Regression  
- **Fitur teks**: TF-IDF  
- **Kelas sentimen**:
  - `0` = Negatif
  - `1` = Positif

## 🔄 Ringkasan Pipeline
1. Ambil dataset Sentiment140.
2. Lakukan cleaning teks (lowercase, hapus URL/mention/karakter non-huruf).
3. Transformasi teks menggunakan TF-IDF.
4. Latih model klasifikasi sentimen.
5. Simpan model dan vectorizer (`.pkl`).
6. Sajikan inferensi melalui aplikasi Streamlit.

## 📝 Catatan
- File model di folder `models/` diperlukan saat menjalankan aplikasi lokal.
- Folder `huggingface_app/` disiapkan untuk deployment terpisah.
- Jika ada perbedaan path model, sesuaikan path di `app/app.py`.

## 📜 Lisensi
Belum ditentukan.  
Disarankan menambahkan file `LICENSE` (mis. MIT) jika repository akan dipublikasikan secara terbuka.