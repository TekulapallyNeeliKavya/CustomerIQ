"""
CustomerIQ - Page 6: Customer Prediction
Classifies new individual customer profiles into optimal clusters using
the fitted StandardScaler and K-Means pipeline.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
)
from src.clustering import predict_single_customer

initialize_app_state()
apply_custom_css()

st.title("Real-Time Customer Segment Prediction")
st.markdown("Input new or prospective customer attributes to determine their assigned behavioral segment and tailored retention playbook.")

df = st.session_state["df_clustered"]
features = st.session_state["features"]
scaler = st.session_state["scaler"]
model = st.session_state["model"]
profiles = st.session_state["profiles"]
centroids_df = st.session_state["centroids_df"]

pipeline_payload = {
    "model": model,
    "scaler": scaler,
    "feature_names": features,
}

# Input Form
with st.form("customer_prediction_form"):
    st.subheader("Customer Feature Inputs")
    input_values = []

    # Responsive grid layout for inputs
    cols = st.columns(min(len(features), 3))
    for i, feat in enumerate(features):
        with cols[i % len(cols)]:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            step_val = 1.0 if max_val - min_val > 10 else 0.1

            val = st.number_input(
                f"{feat}",
                min_value=max(0.0, min_val * 0.5),
                max_value=max_val * 1.5,
                value=round(mean_val, 1),
                step=step_val,
                help=f"Dataset range: {min_val:.1f} to {max_val:.1f} (Average: {mean_val:.1f})",
            )
            input_values.append(val)

    submit_btn = st.form_submit_button("Predict Assigned Customer Segment", type="primary")

if submit_btn:
    # Predict using saved preprocessing and model pipeline
    cluster_idx, scaled_vector = predict_single_customer(input_values, pipeline_payload)

    # Find matching profile
    assigned_profile = next((p for p in profiles if p["cluster_id"] == cluster_idx), None)

    st.success(f"Prediction Complete: Customer Assigned to **Cluster {cluster_idx}**")

    # Display Cluster Profile Card
    if assigned_profile:
        st.markdown(
            f"""
            <div class="cluster-card" style="border-left-color: #2563EB;">
                <div class="cluster-card-header">
                    <div>
                        <span class="cluster-badge" style="background-color: #2563EB20; color: #2563EB;">
                            Cluster {cluster_idx}
                        </span>
                        <span style="font-size: 1.25rem; font-weight: 700; margin-left: 10px; color: #1E293B;">
                            {assigned_profile['segment_name']}
                        </span>
                    </div>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #64748B;">
                        {assigned_profile['archetype_category']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1.2])

        with c1:
            st.markdown("#### Segment Characteristics")
            for char in assigned_profile["important_characteristics"]:
                st.markdown(f"- {char}")

            st.write("")
            st.markdown("#### Segment Benchmarks")
            st.write(f"- **Segment Size**: {assigned_profile['customer_count']} customers ({assigned_profile['percentage']:.1f}% share)")
            if assigned_profile["avg_income"] is not None:
                st.write(f"- **Cluster Avg Income**: ${assigned_profile['avg_income']:.1f}k")
            if assigned_profile["avg_spending"] is not None:
                st.write(f"- **Cluster Avg Spending Score**: {assigned_profile['avg_spending']:.1f}/100")

        with c2:
            st.markdown("#### Tailored Action Playbook for this Customer")
            for rec in assigned_profile["recommendations"]:
                st.markdown(
                    f"""
                    <div class="rec-item">
                        <div class="rec-title">{rec['strategy']}</div>
                        <div class="rec-body"><strong>Action:</strong> {rec['action']}</div>
                        <div class="rec-body" style="color: #64748B;"><strong>Channel:</strong> {rec['channel']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Visual Position Chart if 2 features are active
    if len(features) == 2:
        st.write("")
        st.subheader("Customer Position in Segment Space")
        x_f, y_f = features[0], features[1]

        fig = go.Figure()
        # Add background points
        for c in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == c]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_f],
                    y=sub[y_f],
                    mode="markers",
                    name=f"Cluster {c}",
                    marker=dict(size=7, opacity=0.4),
                )
            )

        # Add centroids
        fig.add_trace(
            go.Scatter(
                x=centroids_df[x_f],
                y=centroids_df[y_f],
                mode="markers+text",
                name="Centroids",
                marker=dict(size=14, symbol="diamond", color="black"),
                text=[f"C{int(c)}" for c in centroids_df["Cluster"]],
                textposition="top center",
            )
        )

        # Add the newly evaluated customer
        fig.add_trace(
            go.Scatter(
                x=[input_values[0]],
                y=[input_values[1]],
                mode="markers+text",
                name="Current Customer",
                marker=dict(size=20, symbol="star", color="#EF4444", line=dict(width=2, color="black")),
                text=["New Customer"],
                textposition="bottom center",
            )
        )

        fig.update_layout(
            title=f"Position Relative to Centroids ({x_f} vs {y_f})",
            xaxis_title=x_f,
            yaxis_title=y_f,
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
