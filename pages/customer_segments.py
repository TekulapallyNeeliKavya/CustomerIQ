"""
CustomerIQ - Page 4: Customer Segment Visualization
Interactive 2D/3D scatter plots, PCA projection for multi-dimensional spaces,
cluster distribution, spending/income patterns, and radar comparison.
"""

import streamlit as st
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
)
from src.visualization import (
    plot_clusters_2d,
    plot_clusters_pca,
    plot_cluster_distribution,
    plot_box_distributions,
    plot_radar_comparison,
)

initialize_app_state()
apply_custom_css()

st.title("Customer Segment Visualization")
st.markdown("Visual behavioral segmentation maps, intra-cluster distributions, and multi-attribute comparative profiles.")

df = st.session_state["df_clustered"]
centroids_df = st.session_state["centroids_df"]
sizes_df = st.session_state["sizes_df"]
features = st.session_state["features"]
scaler = st.session_state["scaler"]

# Section 1: Main Cluster Scatter Plot (2D or PCA)
if len(features) == 2:
    st.subheader(f"2D Customer Clusters: {features[0]} vs {features[1]}")
    fig_scatter = plot_clusters_2d(
        df,
        x_col=features[0],
        y_col=features[1],
        cluster_col="Cluster",
        centroids=centroids_df,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.subheader("High-Dimensional Cluster Visualization (PCA 2D Projection)")
    st.info(
        "**PCA Projection Explanation**: Because the selected feature space contains more than two dimensions "
        f"({', '.join(features)}), Principal Component Analysis (PCA) is applied to project the multidimensional space "
        "onto two orthogonal axes (PC1 and PC2) that capture the maximum variance. Note that axes represent linear "
        "combinations of original features rather than single physical variables."
    )
    fig_pca = plot_clusters_pca(df, features, scaler, cluster_col="Cluster")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Allow custom 2D pair selection
    with st.expander("Explore Specific 2D Feature Pair", expanded=False):
        c_x, c_y = st.columns(2)
        with c_x:
            pair_x = st.selectbox("X-Axis Feature", options=features, index=0)
        with c_y:
            pair_y = st.selectbox("Y-Axis Feature", options=features, index=1 if len(features) > 1 else 0)
        fig_pair = plot_clusters_2d(df, pair_x, pair_y, cluster_col="Cluster", centroids=centroids_df)
        st.plotly_chart(fig_pair, use_container_width=True)

st.write("")

# Section 2: Distribution & Radar Comparison
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("Segment Proportions")
    fig_pie = plot_cluster_distribution(sizes_df)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Multidimensional Segment Radar Profile")
    fig_radar = plot_radar_comparison(centroids_df, features)
    st.plotly_chart(fig_radar, use_container_width=True)

st.write("")

# Section 3: Feature Distribution by Cluster (Boxplots)
st.subheader("Intra-Cluster Feature Distributions")
box_figs = plot_box_distributions(df, features, cluster_col="Cluster")

# Render boxplots in 2 columns
b_cols = st.columns(min(len(box_figs), 2))
for i, f in enumerate(box_figs):
    with b_cols[i % len(b_cols)]:
        st.plotly_chart(f, use_container_width=True)
