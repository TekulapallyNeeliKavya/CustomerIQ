"""
CustomerIQ - Visualization Module
Provides responsive, interactive Plotly visualizations for the Streamlit dashboard,
as well as Matplotlib rendering for static export (PNG).
"""

from typing import List, Optional, Dict, Any
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib


# Color palette for modern executive analytics
PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"]
TEMPLATE = "plotly_white"


def plot_feature_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    """
    Renders an interactive distribution chart with histogram and marginal boxplot.
    """
    fig = px.histogram(
        df,
        x=feature,
        marginal="box",
        nbins=25,
        title=f"Distribution Analysis: {feature}",
        color_discrete_sequence=["#2563EB"],
        template=TEMPLATE,
    )
    fig.update_layout(
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        bargap=0.08,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_features: List[str]) -> go.Figure:
    """
    Renders an interactive correlation matrix heatmap.
    """
    corr = df[numeric_features].corr().round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "black"},
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        template=TEMPLATE,
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(tickangle=-25),
    )
    return fig


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    selected_k: int,
) -> go.Figure:
    """
    Renders the Elbow Method curve with the chosen inflection point highlighted.
    """
    fig = go.Figure()

    # Main curve
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            name="Inertia (SSE)",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=9, color="#3B82F6"),
        )
    )

    # Highlight optimal K
    if selected_k in k_values:
        idx = k_values.index(selected_k)
        fig.add_trace(
            go.Scatter(
                x=[selected_k],
                y=[inertias[idx]],
                mode="markers",
                name=f"Selected Elbow (K={selected_k})",
                marker=dict(size=16, color="#DC2626", symbol="star"),
            )
        )
        fig.add_annotation(
            x=selected_k,
            y=inertias[idx],
            text=f"Selected K={selected_k}<br>SSE={inertias[idx]:.1f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#DC2626",
            ax=40,
            ay=-40,
            bgcolor="#FEF2F2",
            bordercolor="#DC2626",
            borderwidth=1,
        )

    fig.update_layout(
        title="Elbow Method for Optimal K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Sum of Squared Distances)",
        template=TEMPLATE,
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=50, r=40, t=60, b=50),
        xaxis=dict(dtick=1),
    )
    return fig


def plot_silhouette_scores(
    k_values: List[int],
    silhouettes: List[float],
    selected_k: int,
) -> go.Figure:
    """
    Renders Silhouette Score comparison across candidate K values with bar highlighting.
    """
    colors = ["#10B981" if k == selected_k else "#93C5FD" for k in k_values]

    fig = go.Figure(
        data=go.Bar(
            x=k_values,
            y=silhouettes,
            marker_color=colors,
            text=[f"{s:.3f}" for s in silhouettes],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Silhouette Score by Cluster Count",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score (Cohesion vs Separation)",
        template=TEMPLATE,
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=50, r=40, t=60, b=50),
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, max(silhouettes) * 1.25]),
    )
    return fig


def plot_clusters_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "Cluster",
    centroids: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Renders interactive 2D customer segment scatter plot with centroid markers.
    """
    df_plot = df.copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=cluster_col,
        color_discrete_sequence=PALETTE,
        hover_data=[c for c in df.columns if c in ["CustomerID", "Age", "Gender", "Cluster"]],
        title=f"Customer Segments: {x_col} vs {y_col}",
        template=TEMPLATE,
        opacity=0.85,
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))

    # Overlay centroids
    if centroids is not None and x_col in centroids.columns and y_col in centroids.columns:
        fig.add_trace(
            go.Scatter(
                x=centroids[x_col],
                y=centroids[y_col],
                mode="markers+text",
                name="Cluster Centroids",
                marker=dict(size=18, color="#000000", symbol="diamond-open", line=dict(width=3)),
                text=[f"C{int(c)}" for c in centroids["Cluster"]],
                textposition="top center",
                textfont=dict(size=12, color="black", family="Arial Black"),
            )
        )

    fig.update_layout(
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Segment",
    )
    return fig


def plot_clusters_pca(
    df: pd.DataFrame,
    features: List[str],
    scaler: StandardScaler,
    cluster_col: str = "Cluster",
) -> go.Figure:
    """
    Applies Principal Component Analysis (PCA) to project high-dimensional customer feature space
    into 2 orthogonal components for visual inspection.
    """
    X_scaled = scaler.transform(df[features].values.astype(float))
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_ * 100
    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": df[cluster_col].astype(str),
    })
    for col in ["CustomerID", "Age", "Gender"]:
        if col in df.columns:
            pca_df[col] = df[col]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_sequence=PALETTE,
        hover_data=[c for c in pca_df.columns if c not in ["PC1", "PC2"]],
        title=f"Customer Segments (PCA 2D Projection: {var_ratio.sum():.1f}% Variance Explained)",
        labels={
            "PC1": f"Principal Component 1 ({var_ratio[0]:.1f}% Variance)",
            "PC2": f"Principal Component 2 ({var_ratio[1]:.1f}% Variance)",
        },
        template=TEMPLATE,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Segment",
    )
    return fig


def plot_cluster_distribution(sizes_df: pd.DataFrame) -> go.Figure:
    """
    Renders donut chart showing customer share per cluster.
    """
    fig = px.pie(
        sizes_df,
        names=[f"Cluster {c}" for c in sizes_df["Cluster"]],
        values="Customer_Count",
        hole=0.45,
        title="Customer Segment Distribution",
        color_discrete_sequence=PALETTE,
        template=TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_box_distributions(
    df: pd.DataFrame,
    features: List[str],
    cluster_col: str = "Cluster",
) -> List[go.Figure]:
    """
    Renders individual box plots for each feature across clusters.
    """
    figures = []
    df_plot = df.copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)

    for feat in features:
        fig = px.box(
            df_plot,
            x=cluster_col,
            y=feat,
            color=cluster_col,
            color_discrete_sequence=PALETTE,
            title=f"{feat} by Segment",
            template=TEMPLATE,
            points="all",
        )
        fig.update_layout(
            font_family="Inter, system-ui, sans-serif",
            title_font_size=16,
            xaxis_title="Cluster Segment",
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        figures.append(fig)
    return figures


def plot_radar_comparison(
    centroids_df: pd.DataFrame,
    feature_names: List[str],
) -> go.Figure:
    """
    Renders a radar chart comparing normalized centroid profiles across clusters.
    """
    # Normalize values between 0 and 1 for clean radar scaling
    radar_df = centroids_df.copy()
    norm_vals = {}
    for f in feature_names:
        min_v = radar_df[f].min()
        max_v = radar_df[f].max()
        if max_v - min_v > 0:
            norm_vals[f] = (radar_df[f] - min_v) / (max_v - min_v)
        else:
            norm_vals[f] = 0.5

    fig = go.Figure()
    categories = feature_names + [feature_names[0]]

    for i, row in radar_df.iterrows():
        c_id = int(row["Cluster"])
        r_vals = [norm_vals[f][i] for f in feature_names]
        r_vals.append(r_vals[0])  # Close loop

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=categories,
                fill="toself",
                name=f"Cluster {c_id}",
                line=dict(color=PALETTE[c_id % len(PALETTE)]),
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Multidimensional Segment Profile (Normalized Centroids)",
        template=TEMPLATE,
        font_family="Inter, system-ui, sans-serif",
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def save_static_png_charts(
    df: pd.DataFrame,
    features: List[str],
    eval_results: Dict[str, Any],
    selected_k: int,
    output_dir: str,
) -> Dict[str, str]:
    """
    Uses Matplotlib to generate and save static PNG images for reports:
    1. elbow_method.png
    2. silhouette_scores.png
    3. cluster_visualization.png
    """
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # 1. Elbow Method PNG
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eval_results["k_values"], eval_results["inertias"], marker="o", color="#1E3A8A", linewidth=2.2, markersize=7)
    if selected_k in eval_results["k_values"]:
        idx = eval_results["k_values"].index(selected_k)
        ax1.plot(selected_k, eval_results["inertias"][idx], marker="*", color="#DC2626", markersize=16, label=f"Optimal K={selected_k}")
        ax1.legend(loc="upper right", frameon=True)
    ax1.set_title("Elbow Method for Optimal K", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax1.set_ylabel("Inertia (SSE)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "elbow_method.png")
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)
    saved_files["elbow"] = p1

    # 2. Silhouette Scores PNG
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bar_colors = ["#10B981" if k == selected_k else "#93C5FD" for k in eval_results["k_values"]]
    bars = ax2.bar(eval_results["k_values"], eval_results["silhouettes"], color=bar_colors, edgecolor="#1E3A8A", width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Silhouette Score Comparison Across K", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_ylim(0, max(eval_results["silhouettes"]) * 1.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "silhouette_scores.png")
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)
    saved_files["silhouette"] = p2

    # 3. Cluster Visualization PNG
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    if len(features) >= 2:
        x_f, y_f = features[0], features[1]
        scatter = ax3.scatter(
            df[x_f],
            df[y_f],
            c=df["Cluster"],
            cmap="tab10",
            s=55,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
        )
        legend1 = ax3.legend(*scatter.legend_elements(), title="Cluster", loc="best", frameon=True)
        ax3.add_artist(legend1)
        ax3.set_xlabel(x_f, fontsize=11)
        ax3.set_ylabel(y_f, fontsize=11)
        ax3.set_title(f"Customer Clusters: {x_f} vs {y_f}", fontsize=14, fontweight="bold", pad=12)
    ax3.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "cluster_visualization.png")
    fig3.savefig(p3, dpi=300)
    plt.close(fig3)
    saved_files["clusters"] = p3

    return saved_files
