import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model TFLite, scaler, dan label encoder
interpreter = tf.lite.Interpreter(model_path="car_prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Nama Mobil Berdasarkan Spesifikasi")
st.write("Masukkan detail mobil untuk memprediksi nama mobil.")

# Input dari pengguna
year = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2020)
selling_price = st.number_input("Harga Jual Mobil (Rp)", min_value=100000, value=150000000, step=1000000)
fuel = st.selectbox("Jenis Bahan Bakar", options=["Diesel", "Petrol", "CNG", "LPG"])
transmission = st.selectbox("Jenis Transmisi", options=["Manual", "Automatic"])
engine = st.number_input("Kapasitas Mesin (CC)", min_value=500, max_value=5000, value=1500)
seats = st.number_input("Jumlah Kursi", min_value=2, max_value=10, value=5)

# Mapping sesuai encode di notebook
fuel_map = {"Diesel": 0, "Petrol": 1, "CNG": 2, "LPG": 3}
transmission_map = {"Manual": 0, "Automatic": 1}

# Tombol prediksi
if st.button("Prediksi Nama Mobil"):
    input_array = np.array([[year,
                             selling_price,
                             fuel_map[fuel],
                             transmission_map[transmission],
                             engine,
                             seats]])

    input_scaled = scaler.transform(input_array).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output_data)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"Mobil yang diprediksi: **{predicted_label}**")
