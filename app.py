import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")  # Jika menggunakan standard scaler

st.set_page_config(page_title="Dashboard Prediksi IPM", layout="centered")
st.title("📊 Prediksi Indeks Pembangunan Manusia (IPM)")
st.markdown("Masukkan nilai-nilai indikator berikut untuk memprediksi IPM:")

# Sidebar untuk input fitur
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        uhh = st.number_input("Umur Harapan Hidup (UHH)", min_value=0.0)
        hls = st.number_input("Harapan Lama Sekolah (HLS)", min_value=0.0)
    with col2:
        rls = st.number_input("Rata-rata Lama Sekolah (RLS)", min_value=0.0)
        pengeluaran = st.number_input("Pengeluaran Riil per Kapita", min_value=0.0)

    submit = st.form_submit_button("Prediksi")

# Prediksi
if submit:
    input_data = np.array([[uhh, hls, rls, pengeluaran]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    st.success(f"🎯 Hasil Prediksi IPM: {pred[0]:.2f}")
