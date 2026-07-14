import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio


def smart_eda_pro(df: pd.DataFrame, target: str, filename: str = "intelaish_report.html") -> None:
    """
    Compiles an interactive HTML report containing data shape profiles,
    quality audits, and embedded dynamic Plotly visualizations.
    """
    print(f"📁 Compiling futuristic interactive report: {filename}...")

    # 1. Profile Core Variables
    total_rows, total_cols = df.shape
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    # 2. Automatically Generate Plotly Visualizations as Raw HTML Components
    plots_html = ""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    # Generate up to 4 beautiful interactive graphs to embed
    viz_columns = (numeric_cols + categorical_cols)[:4]

    for col in viz_columns:
        if df[col].dtype in [np.number, 'int64', 'float64']:
            fig = px.histogram(df, x=col, color=target, marginal="box",
                               title=f"Distribution Analysis: {col}", template="plotly_dark")
        else:
            fig = px.histogram(df, x=col, color=target, barmode="group",
                               title=f"Categorical Breakdown: {col}", template="plotly_dark")

        # Convert the Plotly figure into raw HTML div string components
        plots_html += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # 3. Construct the HTML Dashboard Template with Cyberpunk/Dark styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>intelaish Automated Data Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0b0f19;
                color: #e2e8f0;
                margin: 0;
                padding: 40px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #00f2fe;
                padding-bottom: 20px;
                margin-bottom: 40px;
            }}
            .header h1 {{
                margin: 0;
                color: #00f2fe;
                font-size: 2.5em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .card {{
                background-color: #171e30;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #2d3748;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            .card h3 {{
                margin: 0 0 10px 0;
                color: #94a3b8;
                font-size: 1em;
            }}
            .card p {{
                margin: 0;
                font-size: 1.8em;
                font-weight: bold;
                color: #38bdf8;
            }}
            .visualizations {{
                display: flex;
                flex-direction: column;
                gap: 40px;
            }}
            .plot-box {{
                background-color: #171e30;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #2d3748;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 intelaish Interactive EDA Report</h1>
                <p>Automated Knowledge Discovery Dashboard</p>
            </div>
            
            <div class="metrics-grid">
                <div class="card"><h3>Total Rows</h3><p>{total_rows}</p></div>
                <div class="card"><h3>Total Columns</h3><p>{total_cols}</p></div>
                <div class="card"><h3>Target Metric</h3><p style="color: #f43f5e;">{target}</p></div>
                <div class="card"><h3>Missing Cells</h3><p style="color: {'#10b981' if missing_count == 0 else '#f59e0b'};">{missing_count}</p></div>
                <div class="card"><h3>Duplicate Rows</h3><p>{duplicate_count}</p></div>
            </div>
            
            <h2 style="color: #00f2fe; margin-bottom: 20px;">📊 Interactive Feature Explorer</h2>
            <div class="visualizations">
                {plots_html}
            </div>
        </div>
    </body>
    </html>
    """

    # 4. Write the file out to the root workspace folder
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(
        f"✨ Report generated successfully! Saved to: {os.path.abspath(filename)}")
