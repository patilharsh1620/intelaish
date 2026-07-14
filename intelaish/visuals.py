import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans


def smart_viz(df: pd.DataFrame, mode: str = "auto", columns: list = None, target: str = None, interactive: bool = True):
    """
    Futuristic visualization engine. Automatically selects the best plot styles,
    creates rotatable 3D scatter plots with K-Means clustering, and builds sleek 2D visuals.
    """
    print(f"🎨 Generating dynamic visualization (Mode: {mode})...")

    # 1. AUTOMATIC MODE: AI-style auto-selection of columns
    if mode == "auto":
        # Separate numeric and categorical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        # If we have 3 or more numeric features, trigger a 3D clustered scatter plot automatically
        if len(numeric_cols) >= 3:
            print(
                "🌊 Multi-dimensional continuous data detected. Building 3D Cluster Environment...")
            plot_3d_scatter_clusters(
                df, columns=numeric_cols[:3], n_clusters=3)
            return
        else:
            # Fallback to standard 2D analysis features
            columns = df.columns.tolist()[:4]

    # 2. MANUAL MODE OR FALLBACK: Generate beautiful, interactive 2D graphs
    if columns:
        for col in columns:
            if df[col].dtype in [np.number]:
                # Dynamic distribution rendering
                fig = px.histogram(df, x=col, color=target, marginal="box",
                                   title=f"Distribution of {col}", template="plotly_dark")
                fig.show()
            else:
                # Dynamic categorical tracking
                fig = px.histogram(df, x=col, color=target, barmode="group",
                                   title=f"Categorical Analysis: {col}", template="plotly_dark")
                fig.show()


def plot_3d_scatter_clusters(df: pd.DataFrame, columns: list, n_clusters: int = 3):
    """
    Runs K-Means clustering under the hood and renders a spectacular 3D scatter environment.
    """
    if len(columns) < 3:
        raise ValueError("3D rendering requires exactly 3 numeric columns.")

    x_col, y_col, z_col = columns[0], columns[1], columns[2]

    # Drop rows with missing values for clustering safety
    cluster_data = df[columns].dropna()

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(cluster_data)

    # Map back labels to visual dataframe
    plot_df = cluster_data.copy()
    plot_df['Cluster'] = [f"Cluster {i}" for i in cluster_labels]

    # Build the 3D Plotly visual object
    fig = px.scatter_3d(
        plot_df, x=x_col, y=y_col, z=z_col,
        color='Cluster',
        title=f"🤖 Advanced 3D Spatial Clustering Engine ({x_col} vs {y_col} vs {z_col})",
        template="plotly_dark",
        opacity=0.8
    )

    # Make visual adjustments to styling
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.show()
