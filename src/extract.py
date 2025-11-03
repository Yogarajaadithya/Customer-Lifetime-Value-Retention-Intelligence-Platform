"""
Data extraction helpers for the CLV & churn prediction platform.

This module exposes a single entry point, ``load_raw_data``, that reads the
daily customer churn feed into a pandas DataFrame ready for downstream
transformations.
"""

from pathlib import Path
from typing import Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_dataset-training-master.csv"


def load_raw_data(path: Union[str, Path] = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset into memory.

    Parameters
    ----------
    path:
        Location of the raw CSV file. Defaults to the canonical dataset
        stored under ``data/raw`` inside the repository.

    Returns
    -------
    pandas.DataFrame
        Untouched representation of the raw churn feed.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}")

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
