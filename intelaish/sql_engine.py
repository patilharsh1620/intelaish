import pandas as pd
from google import genai
from rich.console import Console
from rich.panel import Panel

console = Console()


class SQLBridge:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.ai_enabled = True
            except Exception:
                self.ai_enabled = False
        else:
            self.ai_enabled = False

    def generate_schema(self, df: pd.DataFrame, table_name: str = "dataset") -> str:
        """Translates a Pandas DataFrame into a SQL CREATE TABLE schema."""
        schema = f"CREATE TABLE {table_name} (\n"
        for col, dtype in df.dtypes.items():
            sql_type = "FLOAT" if dtype in [
                'float64'] else "INT" if dtype in ['int64'] else "VARCHAR(255)"
            schema += f"    {col} {sql_type},\n"
        schema = schema.rstrip(",\n") + "\n);"
        return schema

    def text_to_sql(self, df: pd.DataFrame, question: str, table_name: str = "dataset"):
        """Uses AI to translate a plain English question into a SQL query based on the dataframe schema."""
        console.print(
            f"\n[bold blue]🗄️ Translating Question to SQL...[/bold blue]")
        schema = self.generate_schema(df, table_name)

        if not self.ai_enabled:
            console.print(
                "[bold red]❌ AI not enabled. Please provide a Gemini API key to use the SQL Bridge.[/bold red]")
            return

        prompt = f"Given the following SQL schema:\n{schema}\n\nWrite a SQL query to answer this question: '{question}'. Return ONLY the raw SQL code. Do not include markdown formatting or explanations."

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt)
            sql_query = response.text.strip().replace("```sql", "").replace("```", "")

            console.print(Panel(
                f"[bold green]Question:[/bold green] {question}\n\n[bold cyan]Generated SQL Query:[/bold cyan]\n{sql_query}",
                title="🤖 intelguruai SQL Bridge",
                border_style="blue",
                expand=False
            ))
        except Exception as e:
            console.print(f"[bold red]❌ SQL Generation failed: {e}[/bold red]")
