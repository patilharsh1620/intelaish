import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


def smart_read(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file, prints a beautiful summary to the console, 
    and returns a pandas DataFrame.
    """
    console.print(
        f"[bold green]⏳ Loading dataset from:[/bold green] {file_path}")

    try:
        # Load the data
        df = pd.read_csv(file_path)

        # Print a beautiful confirmation panel
        console.print(Panel(
            f"[bold green]✅ Successfully Loaded![/bold green]\n\n"
            f"📊 [bold]Rows:[/bold] {df.shape[0]} | [bold]Columns:[/bold] {df.shape[1]}\n"
            f"💾 [bold]Memory Usage:[/bold] {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            title="intelguruai Ingestion Engine",
            expand=False
        ))

        return df

    except Exception as e:
        console.print(f"[bold red]❌ Error loading file:[/bold red] {str(e)}")
        raise e
