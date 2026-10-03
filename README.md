# 🌍 Global Gini Coefficient Analysis & Forecasting



![Python](https://img.shields.io/badge/Python-3.9%2B-blue)




![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)




![Plotly](https://img.shields.io/badge/Plotly-Visualization-informational)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



End-to-end data science project that cleans, explores and forecasts **country-level income inequality (Gini coefficient, 1963–2025)** using Python, Plotly and scikit-learn.

---

## 📌 Project Overview

The Gini coefficient measures income inequality on a scale from **0** (perfect equality) to **1** (maximum inequality). This project:

- Diagnoses and repairs a corrupted numeric column in the raw dataset
- Explores global and country-level inequality trends with interactive visualizations
- Engineers time-series features (lags, rolling averages, changes, entity trends)
- Trains and compares regression models using a **time-based train/test split**
- Benchmarks every model against a **naive "last value" baseline**
- Exports the trained model and a reusable prediction function

---

## 📊 Dataset

| Property | Value |
|---|---|
| Observations | 2,389 |
| Entities (countries / regions) | 183 |
| Period | 1963 – 2025 |
| Columns | `Entity`, `Code`, `Year`, `Gini coefficient`, `GC Percentage` |

**Data quality issues found and fixed**

- **`Gini coefficient`** was exported with `.` used as a thousands separator (e.g. `3.672.647.476.196.280`). The digits are the decimal part of a 0–1 value, so this reads as **0.3672**. After repair, all values fall in a valid range of 0.177 – 0.711.
- **`GC Percentage`** was inconsistent with the Gini values and unreliable. It is kept for diagnostics only and replaced by `Gini × 100`.
- **Sub-national series** (e.g. *China (rural)*, *Argentina (urban)*) have no country code (`-`) and are flagged with `Is_Subnational`.
- **Uneven coverage:** entities are not observed every year, so a "lag" means the previous *available* observation.

---

## 🔬 Methodology

1. **Data inspection & cleaning**: duplicates, ranges, missing values, numeric repair
2. **Exploratory data analysis**: trends over time, top entities, distributions, outliers, country–year heatmap
3. **Feature engineering**: `Gini_Lag_1/2/3`, `Gini_Rolling_3/5`, `Gini_Change`, `Gini_Pct_Change`, `Entity_Avg_Gini`, `Entity_Gini_Trend`, `Years_Since_Start`, `Decade`
4. **Time-based split**: trained on years ≤ 2019 (1,335 rows), tested on later years (289 rows), so no future data leaks into training
5. **Modeling**: Linear Regression, Random Forest, and Random Forest tuned with `GridSearchCV` + `TimeSeriesSplit`
6. **Evaluation**: MAE, MSE, RMSE and R², compared with a naive baseline
7. **Deployment-ready export**: saved model, scaler and prediction function

---

## 📈 Results

Evaluated on unseen later years:

| Model | R² |
|---|---|
| Naive baseline (repeat last value) | 0.940 |
| Random Forest | 0.944 |
| Linear Regression | 0.947 |
| **Tuned Random Forest** | **0.947** |

**Final tuned model:** MAE = 0.0110 · RMSE = 0.0161 · R² = 0.9465

**Best hyperparameters:** `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`

### 🔑 Key Insight

Gini coefficients are highly persistent: the previous observation (`Gini_Lag_1`) accounts for about **92%** of the Random Forest's feature importance. The models improve only slightly on the naive baseline, which shows that inequality changes slowly and that any forecasting model must be judged against a simple persistence benchmark.

---



## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/global-gini-coefficient-analysis.git
cd global-gini-coefficient-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Gini_Coefficient_Measures.ipynb
```

**`requirements.txt`**

```
pandas
numpy
matplotlib
seaborn
plotly
statsmodels
scikit-learn
joblib
jupyter
```

---

## 🔮 Using the Saved Model

```python
import joblib
import pandas as pd

model  = joblib.load("final_random_forest_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

new_data = pd.DataFrame([{
    "Year": 2024, "Years_Since_Start": 61,
    "Gini_Lag_1": 0.41, "Gini_Lag_2": 0.42, "Gini_Lag_3": 0.42,
    "Gini_Rolling_3": 0.417, "Gini_Rolling_5": 0.418
}])

print(model.predict(scaler.transform(new_data))[0])
```

---

## ⚠️ Limitations

- Predictions are most reliable for entities with long, regular histories.
- Sub-national (urban/rural) series and sparsely observed entities add noise.
- The model uses only past Gini values; it does not include economic or policy drivers (GDP, taxation, employment, etc.).

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn · Joblib

---
