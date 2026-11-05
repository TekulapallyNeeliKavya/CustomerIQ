"""
CustomerIQ - Clustering Module
Implements K-Means clustering, evaluation metrics (Elbow & Silhouette),
cluster assignment, centroid inverse transformation, and model persistence via Joblib.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def evaluate_cluster_range(
    X_scaled: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes Inertia (SSE), Silhouette Score, Davies-Bouldin Index,
    and Calinski-Harabasz Index for a range of candidate K values.

    Returns:
        Dict with keys: 'k_values', 'inertias', 'silhouettes', 'davies_bouldin', 'calinski_harabasz'
    """
    k_values = list(range(min_k, max_k + 1))
    inertias = []
    silhouettes = []
    davies_bouldin = []
    calinski = []

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)

        inertias.append(float(km.inertia_))
        sil_score = float(silhouette_score(X_scaled, labels))
        silhouettes.append(sil_score)
        davies_bouldin.append(float(davies_bouldin_score(X_scaled, labels)))
        calinski.append(float(calinski_harabasz_score(X_scaled, labels)))

    # Identify optimal K by highest silhouette score
    max_sil_idx = int(np.argmax(silhouettes))
    optimal_k_sil = k_values[max_sil_idx]

    # Calculate percentage decrease in inertia (Elbow acceleration)
    inertia_drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski,
        "optimal_k_silhouette": optimal_k_sil,
        "best_silhouette_score": silhouettes[max_sil_idx],
        "inertia_drops": inertia_drops,
    }


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    """
    Fits a KMeans model with k-means++ initialization on scaled data.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(X_scaled)
    return kmeans


def assign_clusters(
    df: pd.DataFrame,
    kmeans: KMeans,
    X_scaled: np.ndarray,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Assigns cluster predictions to a copy of the dataframe.
    """
    df_result = df.copy()
    labels = kmeans.predict(X_scaled)
    df_result[cluster_col] = labels
    return df_result


def get_cluster_centroids(
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extracts cluster centroids and inverts scaling to represent values
    in their original physical business units (e.g. k$ and score 1-100).
    """
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    centroids_df.insert(0, "Cluster", range(len(centroids_df)))
    return centroids_df.round(2)


def get_cluster_sizes(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Calculates customer counts and proportions for each cluster.
    """
    counts = df[cluster_col].value_counts().sort_index()
    total = len(df)
    sizes_df = pd.DataFrame({
        "Cluster": counts.index,
        "Customer_Count": counts.values,
        "Percentage": ((counts.values / total) * 100).round(2),
    })
    return sizes_df


def save_pipeline(
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: List[str],
    evaluation_results: Dict[str, Any],
    file_path: str,
) -> str:
    """
    Saves model, scaler, features, and metadata together via Joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    payload = {
        "model": kmeans,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_clusters": kmeans.n_clusters,
        "evaluation_results": evaluation_results,
    }
    joblib.dump(payload, file_path)
    return file_path


def load_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Loads saved model pipeline payload from Joblib file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model pipeline not found at {file_path}")
    return joblib.load(file_path)


def predict_single_customer(
    customer_values: List[float],
    pipeline: Dict[str, Any],
) -> Tuple[int, np.ndarray]:
    """
    Predicts the assigned cluster for a single customer vector.
    Applies the pipeline's exact fitted scaler and KMeans model.

    Returns:
        Tuple[int, np.ndarray]: (Predicted cluster index, Scaled vector)
    """
    scaler = pipeline["scaler"]
    model = pipeline["model"]

    raw_arr = np.array(customer_values).reshape(1, -1)
    scaled_arr = scaler.transform(raw_arr)
    cluster_idx = int(model.predict(scaled_arr)[0])
    return cluster_idx, scaled_arr
