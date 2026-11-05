"""
CustomerIQ - Preprocessing Module
Handles dataset ingestion, cleaning, validation, exploratory summary, and feature scaling.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(source: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset from a file path or file-like buffer (e.g. Streamlit UploadedFile).
    Cleans column names, standardizes common synonyms, and tracks preprocessing metrics.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Audit Report)
    """
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    initial_shape = df.shape

    # Standardize column names (strip whitespace, normalize casing synonyms)
    df.columns = [c.strip() for c in df.columns]
    column_mapping = {}
    for c in df.columns:
        if c.lower() in ["genre", "sex"]:
            column_mapping[c] = "Gender"
        elif c.lower() in ["cust_id", "id", "customer id", "customer_id"]:
            column_mapping[c] = "CustomerID"
        elif "annual" in c.lower() and "income" in c.lower():
            column_mapping[c] = "Annual Income (k$)"
        elif "spending" in c.lower() and "score" in c.lower():
            column_mapping[c] = "Spending Score (1-100)"
        elif c.lower() == "age":
            column_mapping[c] = "Age"

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Detect duplicates
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Detect missing values
    missing_before = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_before.values()))

    # Handle missing values: impute numeric with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    audit = {
        "initial_rows": initial_shape[0],
        "initial_cols": initial_shape[1],
        "final_rows": df.shape[0],
        "final_cols": df.shape[1],
        "duplicates_removed": duplicate_count,
        "missing_values_handled": total_missing,
        "missing_by_column": missing_before,
        "columns": list(df.columns),
    }

    return df, audit


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical columns in the dataset.
    Excludes identifier columns (like CustomerID) from default clustering features.
    """
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        # Ignore ID columns for numeric feature modeling
        if col.lower() in ["customerid", "id", "cust_id", "index"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def get_descriptive_stats(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculates detailed descriptive statistics including mean, std, min, 25%, 50%, 75%, max, skewness.
    """
    target_df = df[features] if features else df.select_dtypes(include=[np.number])
    stats = target_df.describe().T
    stats["skewness"] = target_df.skew()
    stats["kurtosis"] = target_df.kurtosis()
    return stats.round(2)


def scale_features(
    df: pd.DataFrame,
    features: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scales selected features using StandardScaler.
    If an existing scaler is provided, applies transform(); otherwise fits a new scaler.

    Returns:
        Tuple[np.ndarray, StandardScaler]: (Scaled array, fitted scaler)
    """
    if not features:
        raise ValueError("At least one numerical feature must be selected for scaling.")

    X = df[features].values.astype(float)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def save_cleaned_sample(df: pd.DataFrame, output_path: str, n_samples: int = 15) -> str:
    """
    Saves a representative sample of the cleaned dataset to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df = df.head(n_samples)
    sample_df.to_csv(output_path, index=False)
    return output_path
