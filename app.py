# app.py
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
