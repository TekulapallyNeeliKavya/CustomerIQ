"""
CustomerIQ - Unit and Integration Tests for ML Clustering Pipeline
Tests preprocessing, scaling, evaluation, model fitting, centroids,
pipeline persistence, prediction, and dynamic profiling.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_dataset, scale_features, get_feature_types
from src.clustering import (
    evaluate_cluster_range,
    fit_kmeans,
    assign_clusters,
    get_cluster_centroids,
    get_cluster_sizes,
    save_pipeline,
    load_pipeline,
    predict_single_customer,
)
from src.customer_insights import generate_cluster_profiles, export_profiles_to_dataframe


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")


def test_load_dataset():
    """Verify Mall Customers dataset loads cleanly with standardized columns."""
    df, audit = load_dataset(DATA_PATH)
    assert df.shape[0] == 200
    assert df.shape[1] == 5
    assert "Annual Income (k$)" in df.columns
    assert "Spending Score (1-100)" in df.columns
    assert "Gender" in df.columns
    assert df.isnull().sum().sum() == 0
    assert audit["duplicates_removed"] == 0


def test_feature_scaling():
    """Verify StandardScaler normalizes features to mean 0 and std 1."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = scale_features(df, features)

    assert X_scaled.shape == (200, 2)
    np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-7)


def test_evaluate_cluster_range():
    """Verify evaluation across K=2..10 produces valid inertias and silhouettes."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, _ = scale_features(df, features)

    results = evaluate_cluster_range(X_scaled, min_k=2, max_k=10, random_state=42)

    assert len(results["k_values"]) == 9
    assert results["optimal_k_silhouette"] == 5
    assert results["best_silhouette_score"] > 0.50

    # Verify inertia decreases monotonically as K increases
    inertias = results["inertias"]
    for i in range(len(inertias) - 1):
        assert inertias[i] > inertias[i + 1]


def test_fit_and_assignment():
    """Verify KMeans fitting, cluster assignment, and cluster sizes."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = scale_features(df, features)

    km = fit_kmeans(X_scaled, n_clusters=5, random_state=42)
    df_clustered = assign_clusters(df, km, X_scaled)

    assert "Cluster" in df_clustered.columns
    assert set(df_clustered["Cluster"].unique()) == {0, 1, 2, 3, 4}

    sizes = get_cluster_sizes(df_clustered)
    assert sizes["Customer_Count"].sum() == 200
    assert pytest.approx(sizes["Percentage"].sum(), 0.01) == 100.0


def test_centroids_inversion():
    """Verify centroids are correctly inverted back into original domain coordinates."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = scale_features(df, features)

    km = fit_kmeans(X_scaled, n_clusters=5, random_state=42)
    centroids = get_cluster_centroids(km, scaler, features)

    assert centroids.shape == (5, 3)  # Cluster col + 2 features
    assert (centroids["Annual Income (k$)"] >= 15).all()
    assert (centroids["Annual Income (k$)"] <= 140).all()
    assert (centroids["Spending Score (1-100)"] >= 1).all()
    assert (centroids["Spending Score (1-100)"] <= 100).all()


def test_pipeline_persistence_and_prediction():
    """Verify Joblib save/load pipeline and single customer prediction accuracy."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = scale_features(df, features)
    km = fit_kmeans(X_scaled, n_clusters=5, random_state=42)
    eval_res = evaluate_cluster_range(X_scaled, min_k=2, max_k=5, random_state=42)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_pipeline(km, scaler, features, eval_res, tmp_path)
        loaded = load_pipeline(tmp_path)

        assert loaded["n_clusters"] == 5
        assert loaded["feature_names"] == features

        # Predict for a high-income, high-spending customer (e.g. 90k income, 90 score)
        pred_cluster, scaled_vec = predict_single_customer([90.0, 90.0], loaded)
        assert isinstance(pred_cluster, int)
        assert pred_cluster in range(5)

        # Ensure direct model prediction on same vector gives identical result
        direct_scaled = scaler.transform([[90.0, 90.0]])
        direct_pred = int(km.predict(direct_scaled)[0])
        assert pred_cluster == direct_pred
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dynamic_customer_insights():
    """Verify dynamic profile generation creates at least 2 recommendations per cluster."""
    df, _ = load_dataset(DATA_PATH)
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = scale_features(df, features)
    km = fit_kmeans(X_scaled, n_clusters=5, random_state=42)
    df_clustered = assign_clusters(df, km, X_scaled)

    profiles = generate_cluster_profiles(df_clustered, features)
    assert len(profiles) == 5

    for p in profiles:
        assert "cluster_id" in p
        assert "segment_name" in p
        assert "important_characteristics" in p
        assert len(p["important_characteristics"]) >= 2
        assert "recommendations" in p
        assert len(p["recommendations"]) >= 2
        for r in p["recommendations"]:
            assert "strategy" in r
            assert "action" in r
            assert "channel" in r

    profiles_df = export_profiles_to_dataframe(profiles)
    assert profiles_df.shape[0] == 5
    assert "Cluster ID" in profiles_df.columns
    assert "Strategic Recommendations" in profiles_df.columns


def test_synthetic_custom_dataset_handling():
    """Verify preprocessing handles missing values and custom column names gracefully."""
    synthetic_data = pd.DataFrame({
        "CustomerID": [101, 102, 103, 104, 105],
        "Genre": ["Male", "Female", "Female", "Male", "Female"],
        "Age": [25, 45, np.nan, 32, 50],
        "Annual Income (k$)": [30, 80, 50, np.nan, 95],
        "Spending Score (1-100)": [70, 20, 50, 80, 85],
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        synthetic_data.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        clean_df, audit = load_dataset(tmp_path)
        assert clean_df.shape == (5, 5)
        assert clean_df["Gender"].tolist() == ["Male", "Female", "Female", "Male", "Female"]
        assert clean_df.isnull().sum().sum() == 0  # Missing imputed
        assert audit["missing_values_handled"] == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
