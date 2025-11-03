"""
Model training and scoring pipeline for churn + CLV automation.

Running this module will:
1. Transform the raw dataset into model-ready features.
2. Train an XGBoost classifier to predict churn.
3. Persist the trained model and scored customers to disk.
4. Log parameters, metrics, and artifacts to MLflow for lineage tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import mlflow
    import mlflow.sklearn  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency for local execution
    mlflow = None  # type: ignore[assignment]

from .transform import PROJECT_ROOT, transform_customer_features


EXPORT_PATH = PROJECT_ROOT / "exports" / "churn_scored.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"


def train_and_score(
    export_path: Union[str, Path] = EXPORT_PATH,
    model_path: Union[str, Path] = MODEL_PATH,
) -> dict:
    """
    Execute the full churn training pipeline and return summary metrics.

    Parameters
    ----------
    export_path:
        Destination CSV for the scored customer data.
    model_path:
        Location where the trained model will be serialized.

    Returns
    -------
    dict
        Dictionary containing key performance metrics.
    """
    processed_df = transform_customer_features()

    feature_columns = [
        col
        for col in processed_df.columns
        if col not in {"Churn", "customerID"}
    ]
    X = processed_df[feature_columns]
    y = processed_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    # Score the full dataset
    full_proba = model.predict_proba(X)[:, 1]
    scored_df = processed_df[["customerID", "Churn"]].copy()
    scored_df["churn_probability"] = full_proba
    scored_df["churn_prediction"] = (scored_df["churn_probability"] >= 0.5).astype(int)

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(export_path, index=False)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # MLflow experiment logging
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:10]
    top_features = pd.DataFrame(
        {
            "feature": np.array(feature_columns)[top_idx],
            "importance": importance[top_idx],
        }
    )
    importance_path = export_path.parent / "feature_importance_top10.csv"
    top_features.to_csv(importance_path, index=False)

    if mlflow is not None:
        tracking_uri = (MLRUNS_PATH).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("churn_clv")
        with mlflow.start_run(run_name="daily_churn_training"):
            mlflow.log_params(
                {
                    "model": "XGBClassifier",
                    "n_features": len(feature_columns),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(export_path))
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(importance_path))
    else:
        _log_stub(metrics, feature_columns, export_path, importance_path)

    return metrics


def _log_stub(
    metrics: dict,
    feature_columns: list[str],
    export_path: Path,
    importance_path: Path,
) -> None:
    """
    Minimal fallback logging when MLflow is unavailable in the local runtime.

    Creates a timestamped JSON artifact under ``mlruns/manual_logs`` mimicking
    the telemetry that would be recorded by MLflow.
    """
    import json
    from datetime import UTC, datetime

    manual_dir = MLRUNS_PATH / "manual_logs"
    manual_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "n_features": len(feature_columns),
        "export_path": str(export_path),
        "feature_importance_path": str(importance_path),
    }
    outfile = manual_dir / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
    outfile.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    results = train_and_score()
    print("Training metrics:", results)
