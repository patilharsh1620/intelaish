import pandas as pd
import plotly.express as px
from rich.console import Console
from sklearn.cluster import KMeans
import plotly.graph_objects as go

console = Console()


def smart_viz(df: pd.DataFrame, target: str = None):
    """
    Automatically analyzes the dataset columns and generates 
    the most relevant interactive Plotly charts.
    """
    console.print(
        "\n[bold cyan]🎨 Launching Automated Visualization Engine...[/bold cyan]")

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    # 1. Plot Distributions for Numeric Columns
    for col in numeric_cols:
        console.print(
            f"📈 Generating distribution plot for numeric feature: [bold cyan]{col}[/bold cyan]")
        fig = px.histogram(
            df,
            x=col,
            marginal="box",
            title=f"Distribution of {col}",
            template="plotly_dark",
            color_discrete_sequence=['#00ffcc']
        )
        fig.show()

    # 2. Plot Counts for Categorical Columns
    for col in categorical_cols:
        console.print(
            f"📊 Generating bar chart for categorical feature: [bold magenta]{col}[/bold magenta]")
        fig = px.bar(
            df,
            x=col,
            title=f"Value Counts of {col}",
            template="plotly_dark",
            color_discrete_sequence=['#ff007f']  # Fix applied here!
        )
        fig.show()

    # 3. Relationship Plot (If a target column is provided)
    if target and target in df.columns:
        console.print(
            f"🎯 Target variable specified. Generating relationship plots for: [bold yellow]{target}[/bold yellow]")
        for col in numeric_cols:
            if col != target:
                fig = px.scatter(
                    df,
                    x=col,
                    y=target,
                    trendline="ols",
                    title=f"Relationship: {col} vs {target}",
                    template="plotly_dark"
                )
                fig.show()


def advanced_viz(df: pd.DataFrame, n_clusters: int = 3):
    """Generates a 3D interactive scatter plot."""
    console.print(
        "\n[bold cyan]🚀 Generating 3D Cluster Visualizations...[/bold cyan]")
    numeric_df = df.select_dtypes(include=['int64', 'float64']).dropna()

    if numeric_df.shape[1] < 3:
        console.print(
            "[bold red]❌ Need at least 3 numeric columns for 3D visualization.[/bold red]")
        return

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)

    cols = numeric_df.columns
    fig = go.Figure(data=[go.Scatter3d(
        x=numeric_df[cols[0]], y=numeric_df[cols[1]], z=numeric_df[cols[2]],
        mode='markers', marker=dict(size=8, color=clusters, colorscale='Viridis', opacity=0.8)
    )])

    fig.update_layout(title="3D Cluster Visualization", template="plotly_dark")
    console.print(
        "[bold green]✅ 3D Cluster Visualization Generated![/bold green]")
    fig.show()
