import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def smart_clean(df: pd.DataFrame, target: str = None, outlier_strategy: str = "iqr") -> pd.DataFrame:
    """
    Automatically detects and fixes missing values and outliers in a DataFrame.
    Skips outlier detection for the specified target variable to prevent data destruction.
    """
    cleaned_df = df.copy()
    console.print(
        "\n[bold blue]🧹 Starting Intelligent Data Cleaning Engine (Robust IQR)...[/bold blue]")
    actions = []

    for col in cleaned_df.columns:
        # 1. Handle Missing Values (We still want to fill missing target values if they exist)
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

        # 2. Handle Outliers using IQR (SKIP if it is the target variable!)
        if cleaned_df[col].dtype in ['int64', 'float64'] and col != target:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Avoid divide by zero/zero variance issues
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
    else:
        console.print("[bold green]🎉 Dataset is already clean![/bold green]")

    return cleaned_df
