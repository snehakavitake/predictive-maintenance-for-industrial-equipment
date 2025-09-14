# app.py

# diagnostic_load.py — paste into app.py (replace your existing load_model)
import os, joblib, traceback, streamlit as st

MODEL_PATH = 'pm_model_rf_pipeline.joblib'   # adjust if different name

@st.cache_resource
def load_model():
    st.write("Starting model-load diagnostics...")
    # show files in working dir
    files = [(f, os.path.getsize(f)) for f in os.listdir('.') if os.path.isfile(f)]
    st.write("Files in app root (name, size bytes):", files)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please add it to the repo root or download at runtime.")
        # optional: attempt to download automatically if you set MODEL_URL
        raise FileNotFoundError(MODEL_PATH)

    try:
        m = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully.")
        return m
    except Exception as e:
        st.error("Model load failed. See traceback below.")
        tb = traceback.format_exc()
        # Display traceback in the app (also gets recorded to logs)
        st.text(tb)
        # Re-raise so Streamlit shows error state too
        raise
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("Predictive Maintenance — Demo (RandomForest)")

@st.cache_resource
def load_model():
    return joblib.load('pm_model_rf_pipeline.joblib')

model = load_model()

st.markdown("Upload a CSV with the same feature columns used during training (one or more rows).")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.write("Preview:", data.head())
        # Predictions: pipeline includes preprocessing
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)[:,1]
            data['failure_probability'] = probs
        else:
            data['predicted_failure'] = model.predict(data)
        st.write(data.sort_values(by='failure_probability', ascending=False).head(20))
        st.success("Done — review the failure probabilities to prioritise maintenance.")
    except Exception as e:
        st.error(f"Error running prediction: {e}. Make sure the CSV has the same columns as training data.")
else:
    st.info("No file uploaded. You can upload a CSV or press the example button to run a demo.")
    if st.button("Run example (first 5 test rows)"):
        # You can replace this with a sample saved CSV or show instructions
        st.write("Upload a sample CSV from your training set to demo predictions.")

