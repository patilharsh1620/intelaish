import pandas as pd
import json
from google import genai
from rich.console import Console
from rich.panel import Panel

console = Console()


class SmartInsights:
    def __init__(self, api_key: str = None):
        """
        Initializes the AI insight engine. If an API key is provided, 
        it prepares the modern Gemini client. Otherwise, it uses local statistical heuristics.
        """
        self.api_key = api_key
        if api_key:
            try:
                # Upgraded Initialization: Using the new unified SDK client
                self.client = genai.Client(api_key=api_key)
                self.ai_enabled = True
            except Exception as e:
                console.print(
                    f"[bold yellow]⚠️ Failed to initialize Gemini API. Error: {e}[/bold yellow]")
                self.ai_enabled = False
        else:
            self.ai_enabled = False

    def generate_profile(self, df: pd.DataFrame, target_col: str = None) -> str:
        """
        Analyzes the dataset and returns data insights.
        """
        console.print(
            "\n[bold brain]🧠 Extracting Smart Insights...[/bold brain]")

        # 1. Compute Local Statistical Metadata (Preserves Data Privacy)
        summary_stats = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "numeric_columns_analysis": {},
            "target_variable": target_col
        }

        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            summary_stats["numeric_columns_analysis"][col] = {
                "mean": float(df[col].mean()),
                "skewness": float(df[col].skew()),
                "correlation_with_target": float(df[col].corr(df[target_col])) if target_col and target_col in df.columns and col != target_col else None
            }

        # 2. Path A: Generate Contextual AI Insights via LLM
        if self.ai_enabled:
            prompt = f"""
            Act as an elite Data Scientist and ML Engineer. I have calculated summary statistics for a processed dataset:
            {json.dumps(summary_stats, indent=2)}
            
            Provide an executive summary highlighting key data patterns, unexpected distributions, and strategic recommendations for building a predictive model. 
            Keep it clear, impactful, and under 4 sentences. Do not mention JSON or raw code.
            """
            try:
                # Upgraded Generation Method: Accessed via client.models
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                insight_text = response.text.strip()
                self._display_panel(
                    insight_text, title="🤖 AI-Generated Strategic Insights")
                return insight_text
            except Exception as e:
                console.print(
                    f"[bold red]❌ LLM Generation failed: {e}. Defaulting to rule engine.[/bold red]")

        # 3. Path B: Fallback Local Statistical Heuristics (Zero-Cost, Offline)
        fallback_insights = []
        for col, metrics in summary_stats["numeric_columns_analysis"].items():
            if abs(metrics["skewness"]) > 1.0:
                fallback_insights.append(
                    f"• [bold cyan]{col}[/bold cyan] shows high distribution skewness ({metrics['skewness']:.2f}). Consider a log transform.")
            if metrics["correlation_with_target"] and abs(metrics["correlation_with_target"]) > 0.5:
                fallback_insights.append(
                    f"• Strong linear relationship detected between [bold cyan]{col}[/bold cyan] and target ({metrics['correlation_with_target']:.2f}).")

        if not fallback_insights:
            fallback_insights.append(
                "• Columns show highly stable distributions. Data is well-structured for baseline estimators.")

        insight_text = "\n".join(fallback_insights)
        self._display_panel(insight_text, title="📊 Local Statistical Insights")
        return insight_text

    def _display_panel(self, text: str, title: str):
        console.print(Panel(
            text,
            title=title,
            border_style="purple",
            expand=False
        ))
