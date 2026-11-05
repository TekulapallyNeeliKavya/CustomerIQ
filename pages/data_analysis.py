"""
CustomerIQ - Page 2: Customer Data Analysis
Exploratory Data Analysis (EDA), feature distributions, correlation matrix,
and customer CSV upload capability.
"""

import streamlit as st
import pandas as pd
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
    load_custom_dataset,
    reset_to_default_dataset,
)
from src.preprocessing import get_feature_types, get_descriptive_stats
from src.visualization import plot_feature_distribution, plot_correlation_heatmap

initialize_app_state()
apply_custom_css()

st.title("Customer Data Analysis & Ingestion")
st.markdown("Inspect raw data hygiene, distribution characteristics, correlations, or upload your proprietary customer dataset.")

df = st.session_state["df_raw"]
numeric_cols, cat_cols = get_feature_types(df)
audit = st.session_state.get("audit", {})

# Section 1: Upload Custom Dataset
with st.expander("Upload Proprietary Customer Dataset (CSV)", expanded=False):
    st.markdown("Upload any customer CSV file. The system will automatically detect numerical features, cleanse anomalies, and dynamically retrain the segmentation engine.")
    uploaded_file = st.file_uploader("Select Customer CSV File", type=["csv"])

    col_u1, col_u2 = st.columns([1, 1])
    with col_u1:
        if uploaded_file is not None:
            if st.button("Apply & Cluster Custom Dataset", type="primary"):
                try:
                    with st.spinner("Processing dataset, computing scaling, and identifying optimal clusters..."):
                        load_custom_dataset(uploaded_file)
                    st.success("Successfully ingested and clustered custom dataset! All dashboard pages have been updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing uploaded CSV: {e}")
    with col_u2:
        if st.session_state.get("is_custom_dataset", False):
            if st.button("Reset to Default Mall Customers Dataset"):
                reset_to_default_dataset()
                st.info("Reset to default Mall Customers dataset.")
                st.rerun()

st.write("")

# Section 2: Dataset Overview & Hygiene Audit
st.subheader("Dataset Overview & Hygiene Audit")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Records", f"{df.shape[0]:,}")
with m2:
    st.metric("Feature Columns", f"{df.shape[1]}")
with m3:
    st.metric("Missing Values", f"{sum(audit.get('missing_by_column', {}).values())}")
with m4:
    st.metric("Duplicates Handled", f"{audit.get('duplicates_removed', 0)}")

# Dataset Preview
with st.expander("Dataset Preview (Top 25 Rows)", expanded=True):
    st.dataframe(df.head(25), use_container_width=True)

# Section 3: Descriptive Statistics & Missing Value Audit
c1, c2 = st.columns([1.5, 1])

with c1:
    st.subheader("Descriptive Statistics")
    stats_df = get_descriptive_stats(df, numeric_cols)
    st.dataframe(stats_df, use_container_width=True)

with c2:
    st.subheader("Missing Value Audit")
    miss_df = pd.DataFrame({
        "Column": list(df.columns),
        "Null Count": [int(df[c].isnull().sum()) for c in df.columns],
        "Null (%)": [f"{(df[c].isnull().sum() / len(df)) * 100:.1f}%" for c in df.columns],
        "Dtype": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)

st.write("")

# Section 4: Interactive Feature Distribution
st.subheader("Univariate Feature Distributions")
selected_feature = st.selectbox("Select Feature for Distribution Analysis", options=numeric_cols, index=0)
fig_dist = plot_feature_distribution(df, selected_feature)
st.plotly_chart(fig_dist, use_container_width=True)

st.write("")

# Section 5: Correlation Heatmap
st.subheader("Multivariate Correlation Analysis")
if len(numeric_cols) >= 2:
    fig_corr = plot_correlation_heatmap(df, numeric_cols)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("At least two numerical features are required for correlation analysis.")
