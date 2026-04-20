# Deep Learning Experiment Notebook (BiLSTM)  
**Project:** `pba2026-Nexus`  
**Task:** Advanced Sentiment Analysis with Deep Learning (PyTorch)

---

## 1) Objective

Dokumen ini mendeskripsikan eksperimen lanjutan untuk model Deep Learning berbasis **BiLSTM** pada tugas analisis sentimen tweet.

Target yang dipenuhi:

- Menentukan arsitektur DL
- Verifikasi jumlah parameter model ≤ 10 juta
- Implementasi model dengan PyTorch
- Training + pencatatan kurva training/validation loss
- Evaluasi pada data uji
- Menyiapkan aplikasi demo Streamlit untuk model DL
- Menyiapkan folder deploy Hugging Face Spaces (DL)

---

## 2) Arsitektur Model

Arsitektur yang dipilih: **BiLSTM Classifier**

Pipeline utama:

1. Input teks (`clean_text`)
2. Tokenisasi sederhana berbasis spasi
3. Konversi token ke indeks vocab
4. Padding/truncation ke panjang tetap (`max_len`)
5. Embedding layer
6. BiLSTM (`bidirectional=True`)
7. Dropout
8. Fully-connected classification layer (`num_classes=2`)

Konfigurasi eksperimen:

- `embedding_dim = 128`
- `hidden_dim = 128`
- `num_layers = 2`
- `dropout = 0.3`
- `bidirectional = True`
- `num_classes = 2`

---

## 3) Verifikasi Batas Parameter (≤ 10 Juta)

Hasil verifikasi parameter:

- **Trainable parameters:** `3,219,970` (config checker awal)
- Saat training run aktual (vocab efektif): `1,226,370`
- **Maksimum yang diizinkan:** `10,000,000`
- **Status:** ✅ PASS

Catatan:
Jumlah parameter aktual saat training lebih kecil karena ukuran vocab efektif dari data training (`vocab_size=4425`) lebih kecil dari batas maksimum vocab yang disediakan.

---

## 4) Dataset & Split

Dataset yang dipakai:

- File: `data/cleaned_sample.csv`
- Kolom:
  - `clean_text`
  - `sentiment` (`0=negative`, `1=positive`)

Ukuran data setelah proses:

- Total: `9981`

Strategi split (stratified):

- Train: 80% efektif
- Validation: 10%
- Test: 10%

Hasil ukuran run:

- Train: `7983`
- Validation: `999`
- Test: `999`

---

## 5) Training Setup

Framework: **PyTorch**

Hyperparameter utama:

- Epoch: `6`
- Batch size: `64`
- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Loss function: `CrossEntropyLoss`
- Device: CPU (run saat ini)

Command training contoh:

```bash
python scripts/train_bilstm.py --epochs 6 --batch-size 64 --run-name bilstm_sentiment_v1
```

---

## 6) Hasil Training

Ringkasan log per-epoch:

- Epoch 1: train_loss=0.6236 | val_loss=0.5925 | val_acc=0.6957
- Epoch 2: train_loss=0.5134 | val_loss=0.5477 | val_acc=0.7177
- Epoch 3: train_loss=0.4198 | val_loss=0.5634 | val_acc=0.7197
- Epoch 4: train_loss=0.3166 | val_loss=0.6404 | val_acc=0.7257
- Epoch 5: train_loss=0.2208 | val_loss=0.7366 | val_acc=0.7307
- Epoch 6: train_loss=0.1292 | val_loss=0.9170 | val_acc=0.7317

Indikasi:
- Train loss terus turun tajam.
- Val loss mulai naik setelah epoch ke-2/3 → indikasi overfitting.
- Best validation loss sekitar epoch 2.

Artifact kurva loss disimpan di:

- `assets/dl/bilstm_sentiment_v1_loss_curve.png`
- `assets/dl/bilstm_sentiment_v1_history.csv`

---

## 7) Evaluasi pada Data Uji

Command evaluasi:

```bash
python scripts/eval_bilstm.py --output-json models/dl/eval_metrics.json
```

Hasil utama (test set):

- **Accuracy:** `0.6917`
- **Precision:** `0.6890`
- **Recall:** `0.7000`
- **F1-score:** `0.6944`
- **Confusion Matrix:** `[[341, 158], [150, 350]]`

Interpretasi singkat:
- Performa model cukup seimbang antara kelas negatif dan positif.
- Masih ada ruang peningkatan melalui regularisasi, early stopping, scheduler, atau tuning hyperparameter.

---

## 8) Artifact yang Dihasilkan

Di folder `models/dl/`:

- `bilstm_state_dict.pt`
- `bilstm_sentiment.pt`
- `vocab.json`
- `train_config.json`
- `config.json`
- `train_metrics.json`
- `eval_metrics.json`
- `bilstm_sentiment_v1.pt`
- `bilstm_sentiment_v1_config.json`
- `bilstm_sentiment_v1_metrics.json`
- `bilstm_sentiment_v1_vocab.json`

Di folder `assets/dl/`:

- `bilstm_sentiment_v1_history.csv`
- `bilstm_sentiment_v1_loss_curve.png`

---

## 9) Demo Aplikasi (Streamlit)

Aplikasi DL lokal:

- `app/app_dl.py`

Menjalankan lokal:

```bash
streamlit run app/app_dl.py
```

Fitur demo:

- Input teks
- Prediksi sentimen (NEGATIF/POSITIF)
- Confidence score
- Detail preprocessing (cleaned text, token count, sequence length)

---

## 10) Deploy Hugging Face Spaces (DL)

Folder deploy khusus DL:

- `huggingface_app_dl/`

Isi utama:

- `app.py` (demo inference BiLSTM)
- `Dockerfile`
- `requirements.txt`
- `README.md` (metadata Space)
- `bilstm_state_dict.pt`
- `vocab.json`
- `config.json`

Catatan:
Space DL disiapkan dengan `sdk: docker` agar runtime PyTorch + Streamlit lebih stabil.

---

## 11) Next Improvement (Opsional)

Saran peningkatan performa:

1. Tambahkan **early stopping** berbasis validation loss.
2. Coba **scheduler** (`ReduceLROnPlateau` / `CosineAnnealing`).
3. Tuning:
   - `max_len`
   - `embedding_dim`
   - `hidden_dim`
   - dropout
4. Gunakan pretrained embedding (GloVe/FastText) jika dibutuhkan.
5. Tambahkan error analysis untuk sampel salah klasifikasi.

---

## 12) Reproducibility Checklist

- [x] Seed ditetapkan (`42`)
- [x] Split data stratified
- [x] Config training tersimpan
- [x] Vocab tersimpan
- [x] Metrik evaluasi tersimpan
- [x] Kurva training/validation loss tersimpan

---

## 13) Summary

Eksperimen DL BiLSTM berhasil diimplementasikan end-to-end dan memenuhi ketentuan tugas lanjut:

- ✅ Arsitektur DL ditentukan (BiLSTM)
- ✅ Parameter model diverifikasi ≤ 10 juta
- ✅ Implementasi PyTorch
- ✅ Training + kurva loss
- ✅ Evaluasi test set
- ✅ Demo Streamlit untuk model DL
- ✅ Siap untuk deploy Hugging Face Space (DL)
