import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from rich.console import Console

console = Console()


def explain_model(df: pd.DataFrame, target: str, is_classification: bool = True):
    """
    Trains a baseline model and uses SHAP values to explain feature importance,
    acting as a foundation for bias detection and model interpretability.
    """
    console.print(
        f"\n[bold magenta]🔍 Launching Interpretability & Bias Detection Engine...[/bold magenta]")

    # 1. Prepare data (Drop NaNs for the background model)
    df_clean = df.dropna().copy()
    if target not in df_clean.columns:
        console.print(f"[bold red]❌ Target '{target}' not found.[/bold red]")
        return

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    # 2. Encode categorical variables for the model
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object' or str(y.dtype) == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))

    # 3. Train the baseline model
    model = RandomForestClassifier(
        random_state=42) if is_classification else RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # 4. Calculate SHAP values for interpretability
    console.print(
        "[bold cyan]⚙️ Calculating SHAP values for feature importance...[/bold cyan]")
    explainer = shap.TreeExplainer(model)

    # Use a sample if the dataset is massive to save time
    X_sample = X.sample(min(len(X), 500)) if len(X) > 500 else X
    shap_values = explainer.shap_values(X_sample)

    # 5. Render the plot
    console.print("[bold green]✅ SHAP Summary Plot Generated![/bold green]")

    if is_classification and isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)

    plt.title(f"Feature Importance & Bias Check for '{target}'")
    plt.tight_layout()

    # Save the file instead of blocking the terminal
    file_name = f"shap_summary_{target}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    console.print(
        f"[bold green]✅ SHAP Summary Plot saved to your folder as '{file_name}'![/bold green]")
