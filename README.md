# 🤖 intelaish

**Next-Generation Automated EDA, Data Cleaning, and AI Insights**

[![PyPI - Version](https://img.shields.io/pypi/v/intelaish.svg)](https://pypi.org/project/intelaish/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/intelaish.svg)](https://pypi.org/project/intelaish/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

`intelaish` is a powerful, lightweight Python library designed to automate the most time-consuming parts of the Data Science workflow. From intelligent missing value imputation and automated Machine Learning preprocessing to interactive 3D visualizations and HTML report generation, this tool gets your data 100% ready for Machine Learning in just a few lines of code.

---

## 📖 Table of Contents
1. [Core Features](#✨-core-features-v020)
2. [Installation](#📦-installation)
3. [Quick Start Guide](#💻-quick-start-guide)
4. [Detailed Functionality](#🛠️-detailed-functionality)
5. [Built With](#🛠️-built-with)
6. [About the Author](#👨‍💻-about-the-author)
7. [License](#📄-license)

---

## ✨ Core Features (v0.2.0)

*   🧹 **`smart_clean`**: An intelligent data cleaning engine that automatically handles missing values, applies robust IQR outlier capping, and executes automated ML preprocessing (Z-Score standardization and Categorical Label Encoding).
*   🧠 **`problem_card`**: An automated ML profiler that evaluates the target variable to detect the specific machine learning task (Classification vs. Regression), checks for severe class imbalances, calculates a Data Quality Score, and recommends optimal baseline algorithms.
*   📊 **`smart_viz`**: A dynamic visualization engine powered by Plotly that instantly generates interactive, multi-dimensional 3D scatter plots and clustering environments based on your dataset's continuous and categorical variables.
*   🚀 **`smart_eda_pro`**: A comprehensive reporting tool that compiles your dataset metrics, quality audits, and interactive Plotly charts into a sleek, self-contained, dark-mode HTML dashboard.
*   📥 **`smart_read`**: A robust data ingestion module that safely loads various data formats while optimizing memory usage.

---

## 📦 Installation

You can install `intelaish` directly from the Python Package Index (PyPI) using pip:

```bash
pip install intelaish

Note: Ensure you are using Python 3.8 or higher.

*💻 Quick Start Guide*
Here is a complete end-to-end example of how to ingest data, clean it, profile the ML task, visualize the data in 3D, and generate a web dashboard—all in less than 10 lines of code.

import pandas as pd
from intelaish import (
    smart_read, 
    smart_clean, 
    problem_card, 
    smart_viz, 
    smart_eda_pro
)

# 1. Load Data
df = smart_read("your_dataset.csv")

# 2. Automate Cleaning & ML Preprocessing
# Automatically handles missing values, caps outliers, scales numerics, and encodes text
cleaned_df = smart_clean(df, target="TargetColumn", scale_numeric=True, encode_categorical=True)

# 3. Generate an Intelligent ML Profiler Card
problem_card(cleaned_df, target="TargetColumn")

# 4. Generate Interactive 3D Visualizations
smart_viz(cleaned_df, mode="auto", target="TargetColumn")

# 5. Compile a Full Interactive Web Dashboard
smart_eda_pro(cleaned_df, target="TargetColumn", filename="automated_data_report.html")

🛠️ Detailed Functionality
Automated Preprocessing: intelaish removes the need for manual pipeline construction by wrapping complex Scikit-Learn logic into single function calls.

Performance Reporting: The HTML dashboard output allows for easy sharing of data quality metrics with stakeholders, providing a professional edge to your analysis.

Memory Optimization: The ingestion module is optimized to handle large datasets efficiently without overloading system resources.

🛠️ Built With
Pandas - High-performance data manipulation and analysis.

Scikit-Learn - Industry-standard machine learning preprocessing and scaling.

Plotly - Beautiful, interactive, and exportable data visualizations.

Rich - Advanced terminal formatting and UI dashboards.        
👨‍💻 About the Author
Harsh Patil

Data Analyst specializing in Data Science, AI, and Predictive Modeling. Passionate about building automated, scalable solutions for complex data engineering bottlenecks and business optimization.

LinkedIn: Connect with me

GitHub: patilharsh1620

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
