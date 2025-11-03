"""
Feature engineering utilities for churn and CLV modeling.

This module transforms the raw Telco dataset into a model-ready table that
includes RFM-style metrics and tenure-derived attributes. The cleaned dataset
is also persisted to ``data/processed`` for downstream reuse.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .extract import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "customer_churn_clean.csv"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "CustomerID": "customerID",
        "Usage Frequency": "UsageFrequency",
        "Support Calls": "SupportCalls",
        "Payment Delay": "PaymentDelay",
        "Subscription Type": "SubscriptionType",
        "Contract Length": "ContractType",
        "Total Spend": "TotalSpend",
        "Last Interaction": "LastInteraction",
    }
    return df.rename(columns=rename_map)


def transform_customer_features(
    raw_df: Union[pd.DataFrame, None] = None,
    output_path: Union[str, Path] = DEFAULT_PROCESSED_PATH,
    write_to_disk: bool = True,
) -> pd.DataFrame:
    """
    Clean and enrich customer churn data with modeling features.

    Parameters
    ----------
    raw_df:
        Optional DataFrame of raw customer observations. When ``None``, the
        function will call :func:`load_raw_data` to retrieve the default feed.
    output_path:
        Destination CSV for the processed dataset.
    write_to_disk:
        When True, persist the transformed featureset to ``output_path``.

    Returns
    -------
    pandas.DataFrame
        Feature-complete dataset suitable for model training and scoring.
    """
    df = raw_df.copy() if raw_df is not None else load_raw_data()
    df = _rename_columns(df)

    # Basic cleaning
    numeric_columns = [
        "Age",
        "Tenure",
        "UsageFrequency",
        "SupportCalls",
        "PaymentDelay",
        "TotalSpend",
        "LastInteraction",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_columns = ["Gender", "SubscriptionType", "ContractType"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Target standardisation
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].fillna(0).astype(int)

    # RFM-style features
    df["rfm_recency"] = df["LastInteraction"]
    df["rfm_frequency"] = df["UsageFrequency"] + df["SupportCalls"]
    df["rfm_monetary"] = df["TotalSpend"]

    # Tenure-driven features
    df["tenure_years"] = df["Tenure"] / 12.0
    df["tenure_bucket"] = pd.cut(
        df["Tenure"],
        bins=[-np.inf, 12, 36, 60, np.inf],
        labels=["<1y", "1-3y", "3-5y", "5y+"],
    )

    # Payment behavior
    df["payment_late_flag"] = (df["PaymentDelay"] > 0).astype(int)
    df["avg_monthly_spend"] = df["TotalSpend"] / df["Tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["TotalSpend"])

    # One-hot encode remaining categoricals (excluding identifier)
    feature_df = pd.get_dummies(
        df,
        columns=["Gender", "SubscriptionType", "ContractType", "tenure_bucket"],
        drop_first=True,
    )

    if write_to_disk:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)

    return feature_df


if __name__ == "__main__":
    processed = transform_customer_features()
    print(processed.head())
