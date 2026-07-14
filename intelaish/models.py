import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


def problem_card(df: pd.DataFrame, target: str):
    """
    Analyzes the target variable to determine the ML problem type (Classification/Regression),
    checks for class imbalances, and recommends baseline machine learning models.
    """
    console.print(
        f"\n[bold yellow]🎯 Generating ML Problem Card for target: '{target}'[/bold yellow]")

    if target not in df.columns:
        console.print(
            f"[bold red]❌ Error: Target column '{target}' not found in dataset.[/bold red]")
        return

    target_data = df[target]
    unique_vals = target_data.nunique()
    dtype = target_data.dtype

    # 1. Determine Problem Type Heuristics
    problem_type = "Unknown"
    recommended_models = []
    imbalance_warning = ""

    # If it's numeric and has a wide spread of unique values, it's likely Regression
    if dtype in ['int64', 'float64'] and unique_vals > 15:
        problem_type = "Regression 📈"
        recommended_models = [
            "Linear Regression (Baseline)",
            "Random Forest Regressor (Non-linear)",
            "XGBoost Regressor (Advanced)"
        ]
        imbalance_warning = "[bold green]ℹ️ Continuous target detected. Class imbalance does not apply.[/bold green]"

    # Otherwise, it's Classification
    else:
        problem_type = "Classification 🗂️"
        recommended_models = [
            "Logistic Regression (Baseline)",
            "Random Forest Classifier (Robust)",
            "Gradient Boosting / LightGBM (Advanced)"
        ]

        # 2. Check for Class Imbalance (Crucial ML step)
        value_counts = target_data.value_counts(normalize=True)
        min_class_ratio = value_counts.min()

        if min_class_ratio < 0.2:  # If the smallest class makes up less than 20% of data
            imbalance_warning = f"[bold red]⚠️ Warning: Severe Class Imbalance Detected![/bold red]\n   Smallest class is only {min_class_ratio*100:.1f}% of the dataset.\n   Consider SMOTE, oversampling, or class weighting before training."
        else:
            imbalance_warning = f"[bold green]✅ Class distribution is relatively balanced. Safe to proceed.[/bold green]"

    # 3. Build and Display the UI Panel
    report = (
        f"[bold]Target Variable:[/bold] {target}\n"
        f"[bold]Problem Type Detected:[/bold] {problem_type}\n"
        f"[bold]Unique Classes/Values:[/bold] {unique_vals}\n\n"
        f"{imbalance_warning}\n\n"
        f"[bold cyan]🤖 Recommended ML Models:[/bold cyan]\n"
    )

    for i, model in enumerate(recommended_models, 1):
        report += f"  {i}. {model}\n"

    console.print(Panel(
        report,
        title="📊 intelguruai ML Problem Card",
        border_style="yellow",
        expand=False
    ))
