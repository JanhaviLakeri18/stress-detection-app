import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("stress_model.pkl", "rb"))

st.set_page_config(page_title="Stress Detection App", layout="centered")
st.title("💆 Stress Detection App")
st.write("Predict your stress level using environmental and activity data.")

# Input fields
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
step_count = st.number_input("Step Count", min_value=0, max_value=30000, value=5000)

# Predict button
if st.button("🔍 Predict Stress Level"):
    input_data = np.array([[humidity, temperature, step_count]])
    prediction = model.predict(input_data)[0]
    st.success(f"🧠 Predicted Stress Level: **{prediction}**")
