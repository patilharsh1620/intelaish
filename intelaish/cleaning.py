import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler, LabelEncoder

console = Console()


def smart_clean(df: pd.DataFrame, target: str = None, outlier_strategy: str = "iqr",
                scale_numeric: bool = False, encode_categorical: bool = False) -> pd.DataFrame:
    """
    Automatically detects and fixes missing values and outliers in a DataFrame.
    Skips outlier detection for the specified target variable to prevent data destruction.
    Optionally automates ML preprocessing (Categorical Encoding and Feature Scaling).
    """
    cleaned_df = df.copy()
    console.print(
        "\n[bold blue]🧹 Starting Intelligent Data Cleaning Engine (Robust IQR)...[/bold blue]")
    actions = []

    # --- 1. EXISTING CLEANING LOGIC (Missing Values & Outliers) ---
    for col in cleaned_df.columns:
        # Handle Missing Values
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count > 0:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                actions.append({"Column": col, "Issue": "Missing Values",
                               "Action": f"Filled {missing_count} rows with Median ({fill_val})"})
            else:
                fill_val = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                actions.append({"Column": col, "Issue": "Missing Values",
                               "Action": f"Filled {missing_count} rows with Mode ('{fill_val}')"})

        # Handle Outliers using IQR (SKIP if it is the target variable!)
        if cleaned_df[col].dtype in ['int64', 'float64'] and col != target:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (cleaned_df[col] < lower_bound) | (
                    cleaned_df[col] > upper_bound)
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    cleaned_df[col] = np.clip(
                        cleaned_df[col], lower_bound, upper_bound)
                    actions.append({"Column": col, "Issue": "Outliers Detected",
                                   "Action": f"Capped {outlier_count} values using IQR Bounds"})

    # --- 2. NEW LOGIC: Automated Preprocessing (Categorical Encoding & Feature Scaling) ---
    # Separate remaining columns by data type (excluding target)
    numeric_cols = cleaned_df.select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(
        exclude=[np.number]).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    # Encode Text Categories
    if encode_categorical and len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
            actions.append({"Column": col, "Issue": "Text/Categorical Feature",
                           "Action": "Encoded text labels to numerical tokens"})

    # Scale Numeric Scales
    if scale_numeric and len(numeric_cols) > 0:
        scaler = StandardScaler()
        cleaned_df[numeric_cols] = scaler.fit_transform(
            cleaned_df[numeric_cols])
        for col in numeric_cols:
            actions.append({"Column": col, "Issue": "Unscaled Continuous Range",
                           "Action": "Standardized via Z-Score scaling"})

    # Print summary layout
    if actions:
        table = Table(title="✨ intelguruai Cleaning Summary",
                      show_header=True, header_style="bold magenta")
        table.add_column("Column Name", style="cyan")
        table.add_column("Issue Found", style="yellow")
        table.add_column("Action Taken", style="green")

        for act in actions:
            table.add_row(act["Column"], act["Issue"], act["Action"])
        console.print(table)
        console.print(
            "[bold green]✅ Data is clean and 100% ready for Machine Learning![/bold green]")
    else:
        console.print("[bold green]🎉 Dataset is already clean![/bold green]")

    return cleaned_df
