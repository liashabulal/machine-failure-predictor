import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Load model and scaler ────────────────────
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ── Page config ──────────────────────────────
st.title("🔧 Machine Failure Predictor")
st.write("Enter sensor readings below to predict if a machine will fail.")

# ── Sidebar inputs ────────────────────────────
st.sidebar.header("Sensor Readings")

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=300.0
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=315.0,
    value=310.0
)

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    min_value=1168,
    max_value=2886,
    value=1500
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=3.8,
    max_value=76.6,
    value=40.0
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    min_value=0,
    max_value=253,
    value=100
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    options=[0, 1, 2],
    format_func=lambda x: ["Low (L)", "Medium (M)", "High (H)"][x]
)

# ── Predict button ────────────────────────────
if st.button("Predict"):

    # ── Scale numeric features the same way as training ──
    numeric_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # Raw input dataframe
    input_data = pd.DataFrame([[
        machine_type, air_temp, process_temp, rpm, torque, tool_wear
    ]], columns=[
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ])

    # Apply scaler to only numeric columns
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Get probability
    probability = model.predict_proba(input_data)[0][1]

    # Apply our best threshold from Day 5
    prediction = 1 if probability >= 0.30 else 0

    # ── Show result ───────────────────────────
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ FAILURE PREDICTED — Probability: {probability:.2%}")
    else:
        st.success(f"✅ NORMAL — Probability of failure: {probability:.2%}")

  
# ── SHAP explanation ──────────────────────
    st.subheader("What drove this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Get SHAP values for failure class
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build a simple bar chart
    feature_names = input_data.columns.tolist()
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': sv
    }).sort_values('SHAP Value')

    colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value (red = pushes toward failure, blue = pushes toward normal)')
    ax.set_title('Feature Contribution to Prediction')
    st.pyplot(fig)
