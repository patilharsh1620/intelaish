import pandas as pd
import numpy as np
import plotly.io as pio
from intelaish import (
    smart_read,
    smart_clean,
    problem_card,
    explain_model,
    SQLBridge,
    smart_viz,
    smart_eda_pro  # <-- 1. Added the report function import here
)

pio.renderers.default = "browser"

# 1. Load data
df = smart_read("data.csv")

# 2. Run data cleaning
cleaned_df = smart_clean(
    df, target="Churn", scale_numeric=True, encode_categorical=True)

# 3. Generate ML Problem Card
problem_card(cleaned_df, target="Churn")

# --- ADD A 3RD NUMERIC COLUMN FOR TESTING THE 3D ENGINE ---
# This adds a random column of values between 1 and 10
cleaned_df["Experience"] = np.random.randint(1, 10, size=len(cleaned_df))
# ----------------------------------------------------------

# 4. RUN YOUR FUTURISTIC 3D GRAPH ENGINE!
smart_viz(cleaned_df, mode="auto", target="Churn")

# 5. COMPILE THE FULL INTERACTIVE WEB DASHBOARD!
# <-- 2. Added this step to generate your HTML report file
smart_eda_pro(cleaned_df, target="Churn", filename="my_data_dashboard.html")
