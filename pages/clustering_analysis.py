"""
CustomerIQ - Page 3: Clustering Analysis
Mathematical diagnostics: Elbow Method, Silhouette Analysis,
Cluster Centroids, Sizes, Model Configuration, and justification.
"""

import streamlit as st
import pandas as pd
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
    retrain_model,
)
from src.evaluation import generate_evaluation_table, generate_k_justification
from src.visualization import plot_elbow_curve, plot_silhouette_scores

initialize_app_state()
apply_custom_css()

st.title("K-Means Clustering Diagnostics & Evaluation")
st.markdown("Rigorous mathematical justification for cluster selection using the Elbow Method (SSE) and Silhouette Cohesion Analysis.")

df = st.session_state["df_clustered"]
eval_results = st.session_state["eval_results"]
optimal_k = st.session_state["optimal_k"]
centroids_df = st.session_state["centroids_df"]
sizes_df = st.session_state["sizes_df"]
features = st.session_state["features"]
model = st.session_state["model"]

# Section 1: Interactive Tuning Control
with st.sidebar.expander("Clustering Hyperparameters", expanded=True):
    st.markdown("**Active Features:**")
    st.write(", ".join(features))

    new_k = st.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=int(optimal_k),
        step=1,
        help="Adjust K to evaluate alternative cluster partitioning.",
    )

    if new_k != optimal_k:
        if st.button("Apply Selected K", type="primary"):
            with st.spinner(f"Re-clustering dataset into {new_k} segments..."):
                retrain_model(new_k)
            st.success(f"Model successfully updated to K={new_k}!")
            st.rerun()

# Section 2: Mathematical Evaluation Graphs
col1, col2 = st.columns(2)

with col1:
    fig_elbow = plot_elbow_curve(
        eval_results["k_values"],
        eval_results["inertias"],
        optimal_k,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = plot_silhouette_scores(
        eval_results["k_values"],
        eval_results["silhouettes"],
        optimal_k,
    )
    st.plotly_chart(fig_sil, use_container_width=True)

# Section 3: Justification for Cluster Selection
st.subheader("Mathematical Justification of Optimal K")
justification = generate_k_justification(eval_results, optimal_k, features)

st.markdown(justification["summary"])

c_just1, c_just2, c_just3 = st.columns(3)
with c_just1:
    st.markdown("##### 1. Elbow Inflection (Compactness)")
    st.info(justification["elbow_explanation"])

with c_just2:
    st.markdown("##### 2. Silhouette Peak (Separation)")
    st.success(justification["silhouette_explanation"])

with c_just3:
    st.markdown("##### 3. Business Actionability")
    st.warning(justification["business_explanation"])

st.write("")

# Section 4: Comprehensive Metrics Table
with st.expander("Detailed Evaluation Metrics Across K = 2..10", expanded=False):
    eval_table = generate_evaluation_table(eval_results)
    st.dataframe(eval_table, use_container_width=True, hide_index=True)
    st.caption("Note: Silhouette Score (-1 to +1, higher is better), Davies-Bouldin Index (lower is better), Calinski-Harabasz Index (higher is better).")

st.write("")

# Section 5: Cluster Centroids & Sizes
c_size, c_cent = st.columns([1, 1.6])

with c_size:
    st.subheader("Segment Sizes")
    st.dataframe(sizes_df, use_container_width=True, hide_index=True)

with c_cent:
    st.subheader("Cluster Centroids (Original Units)")
    st.dataframe(centroids_df, use_container_width=True, hide_index=True)

st.write("")

# Section 6: Model Configuration
with st.expander("Scikit-Learn Model Configuration", expanded=False):
    config_df = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Initialization Method",
            "Number of Clusters (K)",
            "n_init (Random Runs)",
            "Max Iterations",
            "Tolerance (tol)",
            "Random State Seed",
            "Feature Scaler",
        ],
        "Setting": [
            "KMeans (Lloyd's Algorithm)",
            model.init,
            str(model.n_clusters),
            str(model.n_init),
            str(model.max_iter),
            str(model.tol),
            str(model.random_state),
            "StandardScaler (Mean=0, Variance=1)",
        ],
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)
