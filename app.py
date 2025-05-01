import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Load model, scaler, dan data dominan fitur
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
shap_df = pd.read_excel("dominant_feature_summary.xlsx")

st.set_page_config(page_title="Dashboard Prediksi & Analisis IPM", layout="wide")
st.title("📊 Dashboard Prediksi & Analisis IPM")

st.markdown("## Prediksi Indeks Pembangunan Manusia (IPM)")
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

st.markdown("---")
st.markdown("## 🔍 Analisis Faktor Dominan per Daerah")

selected_city = st.selectbox("Pilih Kabupaten/Kota:", shap_df['Kabupaten/Kota'].unique())
city_data = shap_df[shap_df['Kabupaten/Kota'] == selected_city].iloc[0]

st.write(f"**Fitur Dominan:** {city_data['Fitur Dominan']}")
st.write(f"**Nilai SHAP:** {city_data['Nilai SHAP']:.2f}")

if city_data['Nilai SHAP'] > 0:
    st.success("✅ Fitur ini berperan **mendorong** IPM kota tersebut.")
elif city_data['Nilai SHAP'] < 0:
    st.error("⚠️ Fitur ini berperan **menghambat** IPM kota tersebut.")
else:
    st.info("ℹ️ Fitur ini tidak terlalu berpengaruh.")
