import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def problem_card(df: pd.DataFrame, target: str) -> None:
    """
    Intelligent profiling card that analyzes the target variable, 
    detects the ML problem type, and recommends optimal baseline models.
    """
    if target not in df.columns:
        console.print(
            f"[bold red]❌ Error: Target column '{target}' not found in dataset.[/bold red]")
        return

    console.print(
        "\n[bold cyan]🔍 Initializing ML Problem Profiler...[/bold cyan]")

    target_series = df[target]
    unique_vals = target_series.nunique()
    dtype = target_series.dtype

    # 1. Detect Problem Type
    if dtype in ['object', 'category', 'bool'] or unique_vals <= 10:
        if unique_vals == 2:
            problem_type = "Binary Classification"
            models = ["Logistic Regression",
                      "Random Forest Classifier", "XGBoost Classifier"]
        else:
            problem_type = "Multi-class Classification"
            models = ["Random Forest Classifier",
                      "LightGBM Classifier", "CatBoost"]
    else:
        problem_type = "Regression"
        models = ["Linear Regression",
                  "Random Forest Regressor", "XGBoost Regressor"]

    # 2. Calculate Data Quality Score
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    # Base score of 100, deduct for missing data and exact duplicates
    quality_penalty = ((missing_cells / total_cells) * 100) + \
        ((duplicate_rows / len(df)) * 10)
    quality_score = max(0.0, min(100.0, 100.0 - quality_penalty))

    if quality_score > 90:
        score_color = "green"
    elif quality_score > 70:
        score_color = "yellow"
    else:
        score_color = "red"

    # 3. Class Imbalance Check (For Classification)
    imbalance_warning = "N/A (Regression)"
    if "Classification" in problem_type:
        value_counts = target_series.value_counts(normalize=True)
        min_class_ratio = value_counts.min()
        if min_class_ratio < 0.20:
            imbalance_warning = f"[red]⚠️ Severe Imbalance Detected (Minority class is {min_class_ratio:.1%})[/red]"
        else:
            imbalance_warning = "[green]✅ Balanced Distribution[/green]"

    # 4. Render the Output Dashboard
    table = Table(show_header=False, box=None)
    table.add_row("[bold]🎯 Target Variable:[/bold]", f"{target}")
    table.add_row("[bold]🧠 Detected Task:[/bold]",
                  f"[bold magenta]{problem_type}[/bold magenta]")
    table.add_row("[bold]📊 Unique Target Values:[/bold]", f"{unique_vals}")
    table.add_row("[bold]⚖️ Class Distribution:[/bold]", imbalance_warning)
    table.add_row("[bold]✨ Data Quality Score:[/bold]",
                  f"[bold {score_color}]{quality_score:.1f} / 100.0[/bold {score_color}]")

    console.print(Panel(
        table, title="🚀 [bold blue]intelguruai Problem Card[/bold blue]", border_style="cyan", expand=False))

    # Render Model Recommendations
    model_table = Table(title="🤖 Recommended Baseline Models",
                        header_style="bold yellow")
    model_table.add_column("Rank", justify="center", style="cyan")
    model_table.add_column("Algorithm", style="green")

    for idx, model in enumerate(models, 1):
        model_table.add_row(f"#{idx}", model)

    console.print(model_table)
    console.print("\n")
