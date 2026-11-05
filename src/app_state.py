"""
CustomerIQ - Application State & UI Utilities Module
Manages centralized session state, default dataset initialization,
and modern UI/UX design components (CSS styling, metric cards, badges).
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
import streamlit as st

from src.preprocessing import load_dataset, get_feature_types, scale_features
from src.clustering import (
    evaluate_cluster_range,
    fit_kmeans,
    assign_clusters,
    get_cluster_centroids,
    get_cluster_sizes,
    load_pipeline,
    save_pipeline,
)
from src.customer_insights import generate_cluster_profiles, export_profiles_to_dataframe
from src.visualization import save_static_png_charts


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.joblib")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def apply_custom_css():
    """
    Applies custom CSS styling for an executive SaaS dashboard feel.
    """
    st.markdown(
        """
        <style>
        /* Base typography and container */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
            border-color: #CBD5E1;
        }
        .metric-card-accent {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2563EB, #3B82F6, #60A5FA);
        }
        .metric-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 6px;
        }
        .metric-card-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }
        .metric-card-sub {
            font-size: 0.82rem;
            color: #10B981;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Cluster persona cards */
        .cluster-card {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border-left: 6px solid #2563EB;
        }
        .cluster-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .cluster-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Recommendation pill */
        .rec-item {
            background-color: #F1F5F9;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
            border-left: 3px solid #3B82F6;
        }
        .rec-title {
            font-weight: 700;
            color: #1E293B;
            font-size: 0.95rem;
        }
        .rec-body {
            color: #475569;
            font-size: 0.88rem;
            margin-top: 4px;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            color: #FFFFFF;
            padding: 32px 36px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #FFFFFF;
            margin-bottom: 8px;
        }
        .hero-desc {
            font-size: 1.05rem;
            color: #94A3B8;
            max-width: 800px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_app_state():
    """
    Initializes session state with the default Mall Customers dataset and pre-trained model.
    Ensures safe reactive synchronization across all pages.
    """
    if "initialized" in st.session_state and st.session_state["initialized"]:
        return

    # Load default dataset
    df, audit = load_dataset(DEFAULT_DATA_PATH)
    default_features = ["Annual Income (k$)", "Spending Score (1-100)"]

    # Check if pre-trained pipeline exists and can be loaded safely
    pipeline_loaded = False
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            pipeline = load_pipeline(DEFAULT_MODEL_PATH)
            kmeans = pipeline["model"]
            scaler = pipeline["scaler"]
            features = pipeline["feature_names"]
            eval_results = pipeline["evaluation_results"]
            pipeline_loaded = True
        except Exception:
            pipeline_loaded = False

    if not pipeline_loaded:
        # Fallback: Train on the fly seamlessly (0.1s)
        features = default_features
        X_scaled, scaler = scale_features(df, features)
        eval_results = evaluate_cluster_range(X_scaled, min_k=2, max_k=10, random_state=42)
        optimal_k = eval_results["optimal_k_silhouette"]
        kmeans = fit_kmeans(X_scaled, n_clusters=optimal_k, random_state=42)
        try:
            save_pipeline(kmeans, scaler, features, eval_results, DEFAULT_MODEL_PATH)
        except Exception:
            pass

    # Prepare initial clusters
    X_scaled, scaler = scale_features(df, features, scaler=scaler)
    df_clustered = assign_clusters(df, kmeans, X_scaled)
    centroids_df = get_cluster_centroids(kmeans, scaler, features)
    sizes_df = get_cluster_sizes(df_clustered)
    profiles = generate_cluster_profiles(df_clustered, features)

    # Store into session state
    st.session_state["df_raw"] = df
    st.session_state["df_clustered"] = df_clustered
    st.session_state["features"] = features
    st.session_state["scaler"] = scaler
    st.session_state["model"] = kmeans
    st.session_state["eval_results"] = eval_results
    st.session_state["optimal_k"] = kmeans.n_clusters
    st.session_state["centroids_df"] = centroids_df
    st.session_state["sizes_df"] = sizes_df
    st.session_state["profiles"] = profiles
    st.session_state["is_custom_dataset"] = False
    st.session_state["audit"] = audit
    st.session_state["initialized"] = True

    # Ensure pre-rendered static charts exist for reports downloads
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    required_pngs = ["elbow_method.png", "silhouette_scores.png", "cluster_visualization.png"]
    if not all(os.path.exists(os.path.join(OUTPUTS_DIR, f)) for f in required_pngs):
        try:
            save_static_png_charts(df_clustered, features, eval_results, kmeans.n_clusters, OUTPUTS_DIR)
        except Exception:
            pass


def retrain_model(n_clusters: int, features: Optional[List[str]] = None):
    """
    Retrains K-Means with specified cluster count and feature set, updating state.
    """
    df = st.session_state["df_raw"]
    if features is None:
        features = st.session_state["features"]

    X_scaled, scaler = scale_features(df, features)
    eval_results = evaluate_cluster_range(X_scaled, min_k=2, max_k=10, random_state=42)
    kmeans = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=42)
    df_clustered = assign_clusters(df, kmeans, X_scaled)

    centroids_df = get_cluster_centroids(kmeans, scaler, features)
    sizes_df = get_cluster_sizes(df_clustered)
    profiles = generate_cluster_profiles(df_clustered, features)

    # Update state
    st.session_state["features"] = features
    st.session_state["scaler"] = scaler
    st.session_state["model"] = kmeans
    st.session_state["eval_results"] = eval_results
    st.session_state["optimal_k"] = n_clusters
    st.session_state["df_clustered"] = df_clustered
    st.session_state["centroids_df"] = centroids_df
    st.session_state["sizes_df"] = sizes_df
    st.session_state["profiles"] = profiles

    # Regenerate static PNGs
    save_static_png_charts(df_clustered, features, eval_results, n_clusters, OUTPUTS_DIR)


def load_custom_dataset(uploaded_file):
    """
    Ingests user-uploaded CSV, selects numeric features, and computes clustering.
    """
    df, audit = load_dataset(uploaded_file)
    numeric_cols, _ = get_feature_types(df)

    if len(numeric_cols) < 2:
        raise ValueError("Uploaded dataset must contain at least 2 numerical features for customer clustering.")

    # Select default features (prioritize income & spending if found, else first 2 numeric)
    inc = next((c for c in numeric_cols if "income" in c.lower()), None)
    spd = next((c for c in numeric_cols if "spending" in c.lower() or "score" in c.lower()), None)

    if inc and spd:
        chosen_features = [inc, spd]
    else:
        chosen_features = numeric_cols[:2]

    X_scaled, scaler = scale_features(df, chosen_features)
    eval_results = evaluate_cluster_range(X_scaled, min_k=2, max_k=10, random_state=42)
    optimal_k = eval_results["optimal_k_silhouette"]
    kmeans = fit_kmeans(X_scaled, n_clusters=optimal_k, random_state=42)
    df_clustered = assign_clusters(df, kmeans, X_scaled)

    centroids_df = get_cluster_centroids(kmeans, scaler, chosen_features)
    sizes_df = get_cluster_sizes(df_clustered)
    profiles = generate_cluster_profiles(df_clustered, chosen_features)

    # Update session state
    st.session_state["df_raw"] = df
    st.session_state["df_clustered"] = df_clustered
    st.session_state["features"] = chosen_features
    st.session_state["scaler"] = scaler
    st.session_state["model"] = kmeans
    st.session_state["eval_results"] = eval_results
    st.session_state["optimal_k"] = optimal_k
    st.session_state["centroids_df"] = centroids_df
    st.session_state["sizes_df"] = sizes_df
    st.session_state["profiles"] = profiles
    st.session_state["is_custom_dataset"] = True
    st.session_state["audit"] = audit

    # Update static files
    save_static_png_charts(df_clustered, chosen_features, eval_results, optimal_k, OUTPUTS_DIR)


def reset_to_default_dataset():
    """
    Resets the session state back to the original Mall Customers dataset.
    """
    st.session_state["initialized"] = False
    initialize_app_state()


def render_metric_card(title: str, value: Any, subtext: str = "", accent_color: str = "#2563EB"):
    """
    HTML renderer for modern SaaS KPI cards.
    """
    card_html = f"""
    <div class="metric-card">
        <div class="metric-card-accent" style="background: {accent_color};"></div>
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-sub">{subtext}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
