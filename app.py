import pickle

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
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

shap_value = city_data['Nilai SHAP']

if shap_value < 0:
    st.error("⚠️ Fitur ini berperan **menghambat** IPM kota tersebut.")
elif shap_value == 0:
    st.info("ℹ️ Fitur ini tidak terlalu berpengaruh/tidak terlalu signifikan.")
else:
    if shap_value < 2.0:
        st.success("✅ Fitur ini berperan **sedikit mendorong** IPM kota tersebut.")
    elif shap_value < 2.25:
        st.success("✅ Fitur ini berperan **cukup mendorong** IPM kota tersebut.")
    elif shap_value < 2.5:
        st.success("✅ Fitur ini berperan **berperan tinggi mendorong** IPM kota tersebut.")
    else:
        st.success("✅ Fitur ini berperan **sangat mendorong** IPM kota tersebut.")

st.title("📊 Prediksi IPM Kabupaten/Kota (2025 2030)")

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("forecast.csv")
    return df

df_prediksi = load_data()

# Validasi isi DataFrame
if df_prediksi.empty or 'Kabupaten' not in df_prediksi.columns:
    st.error("❌ Data kosong atau kolom 'Kabupaten' tidak ditemukan.")
    st.stop()

# === Dropdown kabupaten ===
kabupaten_list = sorted(df_prediksi["Kabupaten"].dropna().unique())
selected_kab = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)

# === Filter dan validasi data kabupaten terpilih ===
data_kab = df_prediksi[df_prediksi["Kabupaten"] == selected_kab]

if data_kab.empty:
    st.warning(f"Tidak ada data untuk {selected_kab}")
    st.stop()

# === Tabel prediksi ===
st.subheader(f"📄 Data Prediksi IPM: {selected_kab}")
st.dataframe(data_kab, use_container_width=True)

# === Ubah ke format long untuk plot ===
pred_cols = [col for col in data_kab.columns if "Prediksi_" in col]

# Validasi kolom prediksi
if not pred_cols:
    st.error("❌ Kolom prediksi tidak ditemukan.")
    st.stop()

data_long = data_kab.melt(id_vars='Kabupaten', value_vars=pred_cols,
                          var_name='Tahun', value_name='Prediksi_IPM')

# Ubah 'Prediksi_2025' → 2025
data_long['Tahun'] = data_long['Tahun'].str.extract(r'(\d+)').astype(int)

# Validasi isi data_long
if data_long.empty:
    st.error("❌ Data untuk visualisasi kosong.")
    st.stop()

# === Plot ===
st.subheader(f"📈 Visualisasi Prediksi IPM 2025 2030: {selected_kab}")
fig = px.line(
    data_long,
    x='Tahun',
    y='Prediksi_IPM',
    markers=True,
    title=f'Prediksi IPM {selected_kab} (2025 2030)',
    labels={'Tahun': 'Tahun', 'Prediksi_IPM': 'Nilai IPM'}
)
st.plotly_chart(fig, use_container_width=True)





# ----------------------
# Load model dan data
# ----------------------
@st.cache_resource
def load_models_and_data():
    with open("arima_models.pkl", "rb") as f:
        arima_models = pickle.load(f)

    df_timeseries = pd.read_csv("df_timeseries.csv", index_col=0)
    df_timeseries.index = pd.to_datetime(df_timeseries.index)
    df_timeseries.index.freq = 'YS'
    return arima_models, df_timeseries

arima_models, df_timeseries = load_models_and_data()

# ----------------------
# Forecast function
# ----------------------
def forecast_arima_future_smooth(arima_models, df_timeseries, years_ahead=6):
    future_forecasts = {}
    last_year_dt = df_timeseries.index.max()
    future_years_index = pd.date_range(start=last_year_dt, periods=years_ahead + 1, freq='YS')

    for kabupaten, model in arima_models.items():
        try:
            last_historical_value = df_timeseries[kabupaten].iloc[-1]
            forecast_result = model.get_forecast(steps=years_ahead)
            predicted_values_future = forecast_result.predicted_mean
            combined_predictions = pd.Series([last_historical_value])._append(predicted_values_future)
            combined_predictions.index = future_years_index
            future_forecasts[kabupaten] = combined_predictions
        except Exception as e:
            future_forecasts[kabupaten] = pd.Series([np.nan]*(years_ahead + 1), index=future_years_index)
    return pd.DataFrame(future_forecasts)

forecast_df_smooth = forecast_arima_future_smooth(arima_models, df_timeseries, years_ahead=6)

# ----------------------
# Streamlit UI
# ----------------------
st.title("📈 Prediksi IPM Kabupaten/Kota di Indonesia (ARIMA)")
kabupaten_list = df_timeseries.columns.tolist()
kabupaten = st.selectbox("Pilih Kabupaten/Kota:", kabupaten_list)

if kabupaten in df_timeseries.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(df_timeseries.index, df_timeseries[kabupaten], label='Historical', color='blue', marker='.')
    plt.plot(forecast_df_smooth.index, forecast_df_smooth[kabupaten], label='Forecast (2025–2030)', linestyle='--', color='red', marker='o')

    plt.title(f"Forecast ARIMA untuk {kabupaten}")
    plt.xlabel("Tahun")
    plt.ylabel("Nilai")
    plt.ylim(0, 100)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())
else:
    st.warning(f"Data untuk {kabupaten} tidak ditemukan.")
