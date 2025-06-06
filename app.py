import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load trained model
model = pickle.load(open("stress_model.pkl", "rb"))

st.set_page_config(page_title="Stress Detection App", layout="centered")
st.title("💆 Stress Detection App")
st.write("Predict your stress level using environmental and activity data.")

# Input fields
humidity = st.slider("Humidity (%)", 0, 100, 50)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
step_count = st.number_input("Step Count", 0, 30000, 5000)

# Predict button
if st.button("🔍 Predict Stress Level"):
    input_data = np.array([[humidity, temperature, step_count]])
    prediction = model.predict(input_data)[0]
        # Map numeric stress level to label
    stress_labels = {
        1: "🟢 Low Stress",
        2: "🟠 Moderate Stress",
        3: "🔴 High Stress"
    }

    # Display result with meaning
    st.success(f"🧠 Predicted Stress Level: {prediction} - {stress_labels.get(prediction, 'Unknown')}")

    # --- Gaussian Chart Part ---
    st.subheader("📊 Gaussian Distribution of Stress Levels")

    # Simulate a normal distribution around the prediction
    mean = prediction
    std_dev = 1  # you can adjust this based on real model variability

    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    y = norm.pdf(x, mean, std_dev)

    # Plot the curve
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Gaussian Curve", color='blue')
    ax.axvline(prediction, color='red', linestyle='--', label=f'Predicted: {prediction}')
    ax.set_title("Stress Level Prediction Curve")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Probability Density")
    ax.legend()

    st.pyplot(fig)
