import pandas as pd
import numpy as np
from intelaish import (
    smart_read, smart_clean, smart_viz, advanced_viz,
    SmartInsights, problem_card, explain_model, SQLBridge
)

# 1. Create dataset
data = pd.DataFrame({
    'Age': [25, np.nan, 35, 42, 28, 150, 31, 45, 29, 38],
    'Salary': [50000, 65000, np.nan, 82000, 58000, 60000, 71000, 90000, 54000, 77000],
    'Department': ['IT', 'HR', 'Finance', None, 'Marketing', 'IT', 'IT', 'Finance', 'HR', 'Marketing'],
    'Churn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
})
data.to_csv("data.csv", index=False)

# 2. Run Pipeline
df = smart_read("data.csv")
cleaned_df = smart_clean(df, target="Churn")

# 3. Generate ML Problem Card
problem_card(cleaned_df, target="Churn")

# 4. Generate SHAP Model Interpretability
explain_model(cleaned_df, target="Churn", is_classification=True)

# 5. Generate Text-to-SQL (Requires API key to fully execute)
sql_bot = SQLBridge(api_key="YOUR_GEMINI_API_KEY_HERE")
sql_bot.text_to_sql(
    cleaned_df, question="What is the average salary of employees who churned?")
