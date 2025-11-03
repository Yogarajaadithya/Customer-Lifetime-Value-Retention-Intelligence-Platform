from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
REFERENCE_DATA = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
FEATURE_DROP = ["customerID", "Churn", "dataset"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_features() -> List[str]:
    reference_df = pd.read_csv(REFERENCE_DATA)
    return [col for col in reference_df.columns if col not in FEATURE_DROP]


def align_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in FEATURE_DROP:
        if col in aligned.columns:
            aligned = aligned.drop(columns=col)
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    missing = [col for col in aligned.columns if col not in feature_columns]
    if missing:
        aligned = aligned.drop(columns=missing)
    return aligned[feature_columns]


st.set_page_config(page_title="Churn Prediction Uploader", layout="wide")
st.title("Customer Lifetime Value & Retention Intelligence")
st.write(
    "Upload a preprocessed customer batch (matching the columns of `data/processed/churn_clean.csv`) "
    "to score churn risk and review retention signals."
)


uploaded_file = st.file_uploader("Upload processed customer CSV", type=["csv"])

if uploaded_file is None:
    st.info("Awaiting data upload. Export a subset from your data warehouse or reuse `churn_clean.csv` for testing.")
    st.stop()


input_df = pd.read_csv(uploaded_file)
if input_df.empty:
    st.warning("Uploaded file is empty. Please provide customer rows.")
    st.stop()


model = load_model()
feature_columns = load_reference_features()
features = align_features(input_df, feature_columns)
probabilities = model.predict_proba(features)[:, 1]


results = input_df.copy()
results["churn_probability"] = probabilities
results["churn_prediction"] = (results["churn_probability"] >= 0.5).astype(int)


st.subheader("Churn Predictions")
st.dataframe(results[["customerID", "churn_probability", "churn_prediction"]].head(100))


st.subheader("Portfolio Summary")
avg_prob = float(np.mean(probabilities))
high_risk_share = float(np.mean(probabilities >= 0.8))
st.metric("Average churn probability", f"{avg_prob:.2%}")
st.metric("High risk share (>= 0.80)", f"{high_risk_share:.2%}")


if "Contract" in results.columns:
    contract_summary = (
        results.groupby("Contract")["churn_probability"].mean().sort_values(ascending=False)
    )
    st.bar_chart(contract_summary)
else:
    st.info("Add a `Contract` column to view contract-level churn insights.")
