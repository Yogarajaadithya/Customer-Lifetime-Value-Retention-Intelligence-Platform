"""
Airflow DAG orchestrating the daily churn & CLV prediction workflow.

The DAG executes three lightweight Python tasks:
1. Extract raw churn data.
2. Transform it into model-ready features.
3. Train and score the churn model, persisting scores + logging to MLflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.transform import transform_customer_features
from src.train_churn import train_and_score


def extract_task():
    # The transform step handles extraction internally. This stub exists to
    # keep the DAG explicit and future-proof for dedicated extraction logic.
    return "extraction-complete"


def transform_task():
    transform_customer_features(write_to_disk=True)
    return "transform-complete"


def train_task():
    train_and_score()
    return "train-complete"


default_args = {
    "owner": "data-platform",
    "depends_on_past": False,
    "email": [],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="churn_clv_pipeline",
    default_args=default_args,
    description="Daily churn & CLV prediction pipeline feeding Power BI exports.",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["churn", "clv", "automation"],
) as dag:
    extract = PythonOperator(
        task_id="extract_raw_customers",
        python_callable=extract_task,
    )

    transform = PythonOperator(
        task_id="transform_features",
        python_callable=transform_task,
    )

    train_score = PythonOperator(
        task_id="train_and_score_churn",
        python_callable=train_task,
    )

    extract >> transform >> train_score
