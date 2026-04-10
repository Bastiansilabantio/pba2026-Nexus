# Hugging Face Deploy App

Folder ini berisi versi aplikasi yang disiapkan khusus untuk deployment (misalnya ke Hugging Face Spaces / Streamlit Cloud), terpisah dari struktur project utama.

## Isi Folder

- `app.py`  
  Aplikasi Streamlit untuk inferensi sentimen tweet.
- `best_sentiment_model.pkl`  
  Model klasifikasi sentimen terlatih.
- `tfidf_vectorizer.pkl`  
  Vectorizer TF-IDF yang digunakan saat training.
- `requirements.txt`  
  Daftar dependency minimal untuk menjalankan app deploy.

## Cara Menjalankan Secara Lokal

1. Masuk ke folder ini:
   ```bash
   cd huggingface_app
   ```

2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Aktifkan virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

5. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

## Catatan

- `app.py` membaca file model (`.pkl`) dari folder yang sama.
- Jika nama atau lokasi file model diubah, path di `app.py` juga perlu disesuaikan.
- Folder ini sengaja dibuat ringkas untuk kebutuhan deployment agar tidak membawa seluruh isi repository utama.