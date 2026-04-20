import streamlit as st
import pickle
import pandas as pd

st.markdown("""
<style>
/* Metric value (number) */
[data-testid="stMetricValue"] {
    color: white !important;
}

/* Metric label (text like Total Records, Normal, etc.) */
[data-testid="stMetricLabel"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI IDS Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>AI Intrusion Detection Dashboard</h1>
<p style='text-align: center; color: gray;'>Real-time anomaly detection using Machine Learning</p>
""", unsafe_allow_html=True)

st.caption("Built using Python, Machine Learning, and Streamlit")

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))

st.success("Model Loaded Successfully")

try:
    data = pd.read_csv("data.csv")  

    st.subheader("Data Preview")
    st.dataframe(data.head())

    for col in data.columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    if "class" in data.columns:
        data = data.drop("class", axis=1)

    data = data[cols]

    predictions = model.predict(data)
    data["Prediction"] = predictions

    anomaly_count = (data["Prediction"] == 1).sum()
    normal_count = (data["Prediction"] == 0).sum()

    st.subheader("System Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Normal", normal_count)
    col3.metric("Anomalies", anomaly_count)

    st.subheader("Security Alerts")

    if anomaly_count > 0:
        st.error(f"{anomaly_count} suspicious activities detected!")
    else:
        st.success("System Secure")

    # Results
    st.subheader("Detailed Results")
    st.dataframe(data.head(20))

except FileNotFoundError:
    st.error("data.csv not found. Please keep dataset in same folder.")