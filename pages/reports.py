"""
CustomerIQ - Page 7: Report Generation & Data Export
Exports labeled customer datasets, centroids, segment sizes, profiling summaries,
and high-resolution visualization charts.
"""

import os
import streamlit as st
import pandas as pd
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
    OUTPUTS_DIR,
)
from src.customer_insights import export_profiles_to_dataframe

initialize_app_state()
apply_custom_css()

st.title("Report Generation & Data Export")
st.markdown("Download comprehensive analytics tables, model centroids, segmentation profiles, and presentation-ready graphics.")

df = st.session_state["df_clustered"]
centroids_df = st.session_state["centroids_df"]
sizes_df = st.session_state["sizes_df"]
profiles = st.session_state["profiles"]
profiles_df = export_profiles_to_dataframe(profiles)
sample_df = df.head(15)

# Section 1: CSV Reports & Tables
st.subheader("Data & Model Output Downloads (CSV)")

c1, c2 = st.columns(2)

with c1:
    st.markdown("##### 1. Full Labeled Customer Dataset")
    st.caption("Complete customer dataset with assigned cluster membership columns.")
    st.download_button(
        label="Download Labeled Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation_labeled.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.write("")
    st.markdown("##### 2. Cluster Centroids")
    st.caption("Mean coordinate centers for each cluster in original physical units.")
    st.download_button(
        label="Download Cluster Centroids (CSV)",
        data=centroids_df.to_csv(index=False).encode("utf-8"),
        file_name="centroids.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.write("")
    st.markdown("##### 3. Cluster Sizes & Proportions")
    st.caption("Count of customers and percentage share across identified segments.")
    st.download_button(
        label="Download Cluster Sizes (CSV)",
        data=sizes_df.to_csv(index=False).encode("utf-8"),
        file_name="cluster_sizes.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    st.markdown("##### 4. Executive Cluster Profiling Summary")
    st.caption("Strategic personas, archetype classifications, key metrics, and actionable playbooks.")
    st.download_button(
        label="Download Cluster Profiles Summary (CSV)",
        data=profiles_df.to_csv(index=False).encode("utf-8"),
        file_name="cluster_profiles.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.write("")
    st.markdown("##### 5. Cleaned Dataset Sample")
    st.caption("Representative 15-row sample of cleansed and normalized customer data.")
    st.download_button(
        label="Download Cleaned Sample (CSV)",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_customers_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.write("")
st.markdown("---")

# Section 2: Visualizations Export (PNG)
st.subheader("High-Resolution Visualization Charts (PNG)")
st.markdown("Pre-rendered 300 DPI publication-quality charts for slide decks and executive memos.")

png_files = {
    "Elbow Method Curve": ("elbow_method.png", "elbow_method.png"),
    "Silhouette Score Comparison": ("silhouette_scores.png", "silhouette_scores.png"),
    "Cluster Scatter Map": ("cluster_visualization.png", "cluster_visualization.png"),
}

v_cols = st.columns(3)

for idx, (title, (fname, out_name)) in enumerate(png_files.items()):
    with v_cols[idx]:
        file_path = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            st.image(img_bytes, caption=title, use_container_width=True)
            st.download_button(
                label=f"Download {title}",
                data=img_bytes,
                file_name=out_name,
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.warning(f"{fname} not found in outputs directory.")
