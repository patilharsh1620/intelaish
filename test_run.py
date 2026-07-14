import pandas as pd
import numpy as np
import plotly.io as pio
from intelaish import (
    smart_read,
    smart_clean,
    problem_card,
    explain_model,
    SQLBridge,
    smart_viz
)

pio.renderers.default = "browser"

# 1. Load data
df = smart_read("data.csv")

# 2. Run data cleaning
cleaned_df = smart_clean(
    df, target="Churn", scale_numeric=True, encode_categorical=True)

# --- ADD A 3RD NUMERIC COLUMN FOR TESTING THE 3D ENGINE ---
# This adds a random column of values between 1 and 10
cleaned_df["Experience"] = np.random.randint(1, 10, size=len(cleaned_df))
# ----------------------------------------------------------

# 3. RUN YOUR FUTURISTIC 3D GRAPH ENGINE!
smart_viz(cleaned_df, mode="auto", target="Churn")
