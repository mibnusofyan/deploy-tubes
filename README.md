# 📊 Prediksi Indeks Pembangunan Manusia (IPM)

Aplikasi dashboard interaktif untuk memprediksi nilai **Indeks Pembangunan Manusia (IPM)** berdasarkan empat indikator utama:

- Umur Harapan Hidup (UHH)
- Rata-rata Lama Sekolah (RLS)
- Harapan Lama Sekolah (HLS)
- Pengeluaran Riil per Kapita

Aplikasi ini dibuat menggunakan:

- 🐍 Python
- 🚀 Streamlit
- 📦 XGBoost
- 📊 SHAP untuk interpretasi model

---

## 🚀 Fitur

- Input interaktif indikator pembangunan
- Prediksi nilai IPM
- Visualisasi model dengan SHAP (opsional untuk versi pengembangan)
- Desain UI modern (menggunakan Streamlit)

---

## 📁 Struktur File

📦 ipm-predictor/ ├── app.py # Main dashboard app ├── scaler.pkl # Preprocessing (StandardScaler) ├── xgb_model.pkl # Trained XGBoost model ├── requirements.txt # Python dependencies └── README.md # This file

## ▶️ Cara Menjalankan Lokal

1. Clone repositori:
   ```bash
   git clone https://github.com/username/ipm-predictor.git
   cd ipm-predictor
   ```
2. Install dependensi:
   pip install -r requirements.txt

3. Jalankan aplikasi:
   streamlit run app.py

Deployment (Streamlit Cloud)
1. Upload semua file (app.py, *.pkl, dll) ke GitHub

2. Deploy di streamlit.io/cloud

3. Pilih file app.py sebagai entry point

📬 Kontak
Dikembangkan oleh Muhammad Ibnu Sofyan
Email: [muhammadibnusofyan003@gmail.com]