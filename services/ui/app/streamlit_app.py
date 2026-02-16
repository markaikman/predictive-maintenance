import streamlit as st
import pandas as pd
from api_client import (
    list_engines,
    get_series,
    predict_latest,
    get_prediction_logs,
    get_feature_drift,
)

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("Predictive Maintenance — Fleet Dashboard (FD001)")
tab_fleet, tab_monitor = st.tabs(["Fleet", "Monitoring"])


@st.cache_data(ttl=60)
def cached_engines():
    return list_engines()


# Keep UI running if FastAPI not up, provide clear error log.
try:
    engines = cached_engines()
except Exception as e:
    st.error("API is not reachable yet. Is the FastAPI container running?")
    st.exception(e)
    st.stop()

with tab_fleet:
    with st.sidebar:
        st.header("Select Engine")
        engine_id = st.selectbox("Engine ID", engines, index=0)

        st.header("Series to Plot")
        sensor_cols = [f"s{i}" for i in range(1, 22)]
        cols = st.multiselect("Sensors", sensor_cols, default=["s2", "s3", "s4"])

        st.header("Prediction Settings")
        window = st.slider(
            "Rolling window", min_value=3, max_value=50, value=10, step=1
        )

        load = st.button("Load Series")
        run_pred = st.button("Predict Latest Cycle")

    left, right = st.columns([2, 1], gap="large")

    if load:
        rows = get_series(int(engine_id), cols=["cycle"] + cols)
        df = pd.DataFrame(rows)

        with left:
            st.subheader(f"Engine {engine_id} — Sensor Time Series")
            if df.empty:
                st.warning("No data returned.")
            else:
                st.line_chart(df.set_index("cycle")[cols])

        with right:
            st.subheader("Latest Values")
            st.dataframe(df.tail(1))

    if run_pred:
        out = predict_latest(int(engine_id), window=int(window))

        st.subheader("RUL Prediction (Latest Cycle)")
        st.metric("Predicted RUL", f"{out['prediction_rul']:.2f}")
        st.caption(
            f"Engine: {out['engine_id']} | Cycle: {out['cycle']} | Window: {out['window']} | Model: {out['model_version']}"
        )
        st.success("Prediction logged to Postgres (prediction_logs).")
    else:
        st.info("Use the sidebar to **Load Series** and/or **Predict Latest Cycle**.")

with tab_monitor:
    st.subheader("Monitoring")

    colA, colB = st.columns(2)
    with colA:
        limit = st.number_input(
            "Recent predictions to analyze",
            min_value=50,
            max_value=5000,
            value=1000,
            step=50,
        )
    with colB:
        refresh = st.button("Refresh monitoring")

    try:
        logs = get_prediction_logs(int(limit))
        df = pd.DataFrame(logs)

        if df.empty:
            st.warning(
                "No predictions logged yet. Use the Fleet tab to generate some predictions."
            )
        else:
            # Parse created_at timestamps
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

            st.markdown("### Prediction volume")
            vol = (
                df.dropna(subset=["created_at"])
                .set_index("created_at")
                .resample("1min")
                .size()
                .rename("count")
                .to_frame()
            )
            st.line_chart(vol)

            st.markdown("### Prediction distribution")
            st.bar_chart(df["prediction"].astype(float))

            st.markdown("### Feature drift vs training (PSI)")
            drift = get_feature_drift(int(limit))
            if not drift.get("ok"):
                st.warning(drift.get("error", "Drift unavailable"))
            else:
                top = pd.DataFrame(drift["top"])
                st.dataframe(top, use_container_width=True)
                st.caption(
                    "Rule of thumb: PSI < 0.10 ok, 0.10–0.25 investigate, >0.25 high drift."
                )
    except Exception as e:
        st.error("Monitoring API not reachable or returned an error.")
        st.exception(e)
