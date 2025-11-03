# Customer Lifetime Value & Retention Prediction Platform

Beginner-friendly guide to the end-to-end project that automates customer churn prediction, enriches the data with CLV-style features, and delivers daily-scored exports for Business Intelligence (Power BI) dashboards.

---

## 1. What problem are we solving?

Subscription businesses need to know **which customers are likely to churn** (leave) and **how valuable those customers are** so teams can plan retention offers, prioritise outreach, and forecast revenue.  

This project:
1. Ingests the Telco Customer Churn dataset (the CSV is already included under `data/raw/`).
2. Cleans and enriches the data with business-friendly features such as tenure buckets and RFM (recency/frequency/monetary) metrics.
3. Trains a machine-learning model that predicts churn probability.
4. Writes the scored results to `exports/churn_scored.csv` so analysts can visualise it (Power BI).
5. Prepares orchestration (Airflow DAG) and experiment tracking (MLflow, or a fallback) so the process can run automatically every day.

---

## 2. Tech stack overview (why these tools?)

| Tool / Library      | Why we use it                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| **Python 3.11+**    | Primary language for data processing and ML.                                 |
| **pandas / numpy**  | Data wrangling, feature engineering, numeric calculations.                   |
| **scikit-learn**    | Helper functions (train/test split, metrics).                                |
| **XGBoost**         | Gradient boosted trees—performs well on tabular churn data.                  |
| **joblib**          | Serialize trained models to disk (`models/churn_model.pkl`).                 |
| **MLflow (optional)**| Track parameters/metrics/models for each training run. Falls back to JSON logs if MLflow isn’t installed. |
| **Apache Airflow**  | Orchestrates the extract → transform → train workflow on a daily schedule.   |
| **Power BI**        | Business-facing dashboard that consumes the scored CSV export.               |

> **Note on Python version:** Apache Airflow and MLflow do not yet publish wheels for Python 3.14. If you need the full stack, create your virtual environment with Python 3.11 or 3.12.

---

## 3. Project structure (what lives where?)

```
project-root/
├── dags/                      # Airflow DAG (daily automation)
│   └── churn_clv_pipeline.py
├── src/                       # Reusable Python modules
│   ├── extract.py             # Loads the raw CSV
│   ├── transform.py           # Cleans + enriches data (RFM, tenure, encodings)
│   └── train_churn.py         # Trains model, scores customers, logs run
├── data/
│   ├── raw/                   # Input dataset (already provided)
│   └── processed/             # Outputs from the transform & training steps
├── exports/                   # Power BI-ready churn_scored.csv + feature ranks
├── models/                    # Serialized churn model (joblib)
├── mlruns/                    # MLflow tracking data OR JSON fallback logs
├── notebooks/                 # Exploratory analysis & modeling notebooks
├── reports/                   # CSV summaries (from notebooks)
└── requirements.txt           # Project dependencies
```

Empty folders include a `.gitkeep` so they remain in version control until populated.

---

## 4. Data flow from start to finish

### Step 1 – Extraction (`src/extract.py`)
- Function: `load_raw_data()`
- Reads `data/raw/customer_churn_dataset-training-master.csv` into a pandas DataFrame.
- Throws a clear error if the file is missing.

### Step 2 – Transformation (`src/transform.py`)
- Function: `transform_customer_features(...)`
- Cleans numeric columns (e.g., tenure, support calls) by coercing invalid values and filling simple gaps.
- Renames columns into code-friendly names (`Subscription Type` → `SubscriptionType`).
- Calculates new features:
  - **RFM metrics:** recency (`LastInteraction`), frequency (Usage + Support Calls), monetary (`TotalSpend`).
  - **Tenure:** buckets like `<1y`, `1-3y`, `3-5y`, `5y+`.
  - **Payment behaviour:** late-payment flag, average monthly spend.
- One-hot encodes categorical features (gender, subscription type, contract type).
- Saves the full enriched dataset to `data/processed/customer_churn_clean.csv`.

### Step 3 – Training & Scoring (`src/train_churn.py`)
- Function: `train_and_score(...)` (also executed when you run `python -m src.train_churn`).
- Splits the processed dataset into training and validation sets.
- Fits an `XGBClassifier` with sensible defaults (300 trees, moderate learning rate, etc.).
- Evaluates accuracy, ROC-AUC, and F1 score.
- Scores every customer:
  - Output columns: `customerID`, `Churn` (actual), `churn_probability`, `churn_prediction` (0/1 threshold).
  - Saves to `exports/churn_scored.csv`.
- Writes top 10 feature importances to `exports/feature_importance_top10.csv`.
- Saves the trained model to `models/churn_model.pkl`.
- **Run logging:**
  - If MLflow is available, logs metrics, parameters, model artefact, and outputs to an MLflow run under `mlruns/`.
  - If MLflow is not installed (e.g., Python 3.14), logs a JSON summary in `mlruns/manual_logs/`.

### Step 4 – Automation (Airflow DAG)
- File: `dags/churn_clv_pipeline.py`
- Declares three Python tasks: extract → transform → train/score.
- Designed to run daily (`schedule_interval="@daily"`).
- When Airflow is pointed at this repository, enabling the DAG will run the same Python functions automatically and keep exports up to date.

### Step 5 – Visualization (Power BI)
- Power BI connects to `exports/churn_scored.csv`.
- Refresh the dataset daily so business users always see the freshest churn probabilities and driver insights.

---

## 5. Getting started (step-by-step)

### 5.1. Clone the project
```powershell
git clone <your-repo-url>
cd "Customer Lifetime Value & Retention Intelligence Platform"
```

### 5.2. Create a virtual environment
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
```

> If you plan to add Airflow/MLflow, use Python 3.12 or 3.11 when creating the virtual environment.

### 5.3. Install dependencies
```powershell
pip install -r requirements.txt
```

- On Python 3.14, Airflow and MLflow currently fail to install. The pipeline still runs because the code gracefully falls back when MLflow is absent.
- On Python ≤ 3.12, you can optionally run:
  ```powershell
  pip install mlflow
  pip install "apache-airflow==2.9.3" --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt
  ```

### 5.4. Execute the pipeline locally
```powershell
python -m src.train_churn
```

You will see a metrics summary like:
```
Training metrics: {'accuracy': 0.999, 'roc_auc': 0.999996, 'f1': 0.99926}
```

Generated artefacts:
- `data/processed/customer_churn_clean.csv`
- `exports/churn_scored.csv`
- `exports/feature_importance_top10.csv`
- `models/churn_model.pkl`
- `mlruns/` folder populated with logs (`manual_logs/*.json` if MLflow missing).

### 5.5. Wire up Power BI
1. Open Power BI Desktop → **Get Data** → **Text/CSV**.
2. Select `exports/churn_scored.csv`.
3. Load the data and build visuals (e.g., churn probability distributions, top drivers).
4. For scheduled refresh:
   - Upload the CSV to a shared location (OneDrive/SharePoint or network drive).
   - Point Power BI Service Gateway to that path.
   - Schedule the refresh to align with your Airflow (or manual) pipeline run.

---

## 6. How the notebooks fit in

The `notebooks/` directory contains ordered experiments:
- `01_exploration.ipynb`: Quick data audit and initial cleaning ideas.
- `02_modeling.ipynb`: Earlier version of the training workflow; useful for interactive tuning.
- `03_insights.ipynb`: Derives actionable insights (high-risk customers, cohort churn rates).

These notebooks informed the production pipeline you now run through `src/` and Airflow.

---

## 7. Extending or customizing the project

- **Add CLV modelling**: Incorporate lifetime value predictions (e.g., using survival analysis or revenue forecasts) and append them to the export.
- **Switch algorithms**: Experiment with logistic regression, CatBoost, or neural nets by editing `train_churn.py`.
- **Enhance features**: Join CRM data (discounts, loyalty points), compute additional RFM segments, or encode customer segments.
- **Deploy to the cloud**: Containerise the pipeline and run it via AWS MWAA, GCP Composer, or Azure Data Factory with dbt/Databricks.
- **Dashboard enhancements**: Include explainability visualisations (SHAP values) in the export to help stakeholders understand predictions.

---

## 8. Troubleshooting & FAQs

| Issue | Cause & Fix |
|-------|-------------|
| `FileNotFoundError: Raw dataset not found` | Ensure `data/raw/customer_churn_dataset-training-master.csv` exists. Copy it from the original dataset if missing. |
| `ModuleNotFoundError: No module named 'mlflow'` | Install MLflow on Python ≤ 3.12, or ignore (the code falls back to JSON logging). |
| `pip` cannot install Apache Airflow | Airflow does not yet support Python 3.14; recreate the environment on Python 3.11/3.12. |
| Power BI won’t refresh | Confirm the CSV path is reachable by the Power BI Gateway and that the pipeline writes files before the scheduled refresh. |
| Model metrics look too perfect | The dataset is synthetic; real-world data will have noisier metrics. Use cross-validation and consider calibrating probabilities. |

---

## 9. Recap

You now have a fully scaffolded churn + CLV prediction platform that:
1. Loads raw customer churn data.
2. Engineers meaningful business features.
3. Trains a predictive model and scores every customer daily.
4. Logs runs (via MLflow or JSON fallback) for reproducibility.
5. Delivers a Power BI-ready dataset so stakeholders track risk in near real time.

Take the starter code, plug in your own data sources, and expand it to suit your organisation’s retention strategy. Happy building!
