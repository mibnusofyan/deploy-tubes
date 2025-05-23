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

selected_city = st.selectbox("Pilih Kabupaten/Kota:", shap_df['Kabupaten/Kota'].unique(), key="shap_kab_select")
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
selected_kab = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list, key="forecast_kabupaten_select")

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
# Load model dan data (ARIMA)
# ----------------------
@st.cache_resource
def load_models_and_data():
    arima_models = joblib.load("arima_models.pkl")

    df_timeseries = pd.read_csv("df_timeseries.csv", index_col=0)
    df_timeseries.index = pd.to_datetime(df_timeseries.index)
    df_timeseries.index.freq = 'YS'
    return arima_models, df_timeseries

# Now call the function to load the data and models for ARIMA
arima_models, df_timeseries = load_models_and_data()

# ----------------------
# Forecast function
# ----------------------
def forecast_arima_future_smooth(arima_models, df_timeseries, years_ahead=6):
    future_forecasts = {}
    last_year_dt = df_timeseries.index.max()
    # Ensure future_years_index starts *after* the last historical year if you only want future dates
    # Or include the last year if you want a smooth transition point in the forecast plot
    # The current code includes the last historical year in future_years_index and combined_predictions
    future_years_index = pd.date_range(start=last_year_dt, periods=years_ahead + 1, freq='YS')

    for kabupaten, model in arima_models.items():
        try:
            # Get the last historical value for the smooth transition
            last_historical_value = df_timeseries[kabupaten].iloc[-1]

            # Get the forecast from the loaded model
            forecast_result = model.get_forecast(steps=years_ahead)
            predicted_values_future = forecast_result.predicted_mean

            # Combine the last historical value and the future predictions
            # Use pd.concat instead of the deprecated _append
            combined_predictions = pd.concat([pd.Series([last_historical_value]), predicted_values_future])
            combined_predictions.index = future_years_index # Assign the correct future index
            future_forecasts[kabupaten] = combined_predictions
        except Exception as e:
            # Handle cases where forecasting might fail for a specific kabupaten
            print(f"Error forecasting for {kabupaten}: {e}") # Optional: print error for debugging
            future_forecasts[kabupaten] = pd.Series([np.nan]*(years_ahead + 1), index=future_years_index)
    return pd.DataFrame(future_forecasts)

forecast_df_smooth = forecast_arima_future_smooth(arima_models, df_timeseries, years_ahead=6)


# ----------------------
# Streamlit UI
# ----------------------

st.title("📈 Prediksi IPM Kabupaten/Kota di Indonesia (ARIMA)")
kabupaten_list = df_timeseries.columns.tolist()
kabupaten = st.selectbox("Pilih Kabupaten/Kota:", kabupaten_list, key="arima_kab_select")

if kabupaten in df_timeseries.columns:
    fig, ax = plt.subplots(figsize=(10, 5)) # Use object-oriented matplotlib approach
    ax.plot(df_timeseries.index, df_timeseries[kabupaten], label='Historical', color='blue', marker='.')
    ax.plot(forecast_df_smooth.index, forecast_df_smooth[kabupaten], label='Forecast (2025–2030)', linestyle='--', color='red', marker='o')

    ax.set_title(f"Forecast ARIMA untuk {kabupaten}")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Nilai")
    ax.set_ylim(0, 100) # Set appropriate y-limits
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Ensure every year is a major tick
    plt.xticks(rotation=45) # Rotate labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    ax.grid(True)
    ax.legend()
    st.pyplot(fig) # Pass the figure object to st.pyplot
else:
    st.warning(f"Data untuk {kabupaten} tidak ditemukan.")