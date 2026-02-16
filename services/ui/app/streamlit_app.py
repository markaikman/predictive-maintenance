import streamlit as st
import pandas as pd
from api_client import predict

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("Predictive Maintenance — Demo Dashboard")

with st.sidebar:
    st.header("Input")
    engine_id = st.text_input("Engine ID", "Engine_12")

    st.caption(
        "Add a few numeric features for now (we'll replace with real sensor-derived features)."
    )
    f1 = st.number_input("sensor_1", value=100.0)
    f2 = st.number_input("sensor_2", value=0.25)
    f3 = st.number_input("sensor_3", value=42.0)

    run = st.button("Predict")

if run:
    features = {"sensor_1": f1, "sensor_2": f2, "sensor_3": f3}
    out = predict(engine_id, features)

    st.subheader("Prediction")
    st.metric("Predicted risk / RUL proxy", f"{out['prediction']:.4f}")
    st.caption(f"Model version: {out['model_version']}")

    st.subheader("Features")
    st.dataframe(pd.DataFrame([features]))
else:
    st.info("Enter features in the sidebar and click **Predict**.")
