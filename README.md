# AI-ML-Challenge2-IITMadras
# 🌡️ Multi-Station Weather Temperature Prediction

> **Shaastra Techathon AI/ML Challenge 2** — IIT Madras  
> Predicting average temperature (TAVG) from multi-station meteorological data using ensemble learning.

---

## 📌 Problem Statement

Given historical weather observations from **three geographically distributed stations (A, B, C)**, predict the **composite average temperature (TAVG)** for a target date. Each station records precipitation, snow depth, maximum temperature, minimum temperature, and average temperature — resulting in a 15-feature input space.

---

## 📂 Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 812 | `train.csv` |
| Test  | 203 | `test.csv` |

**Features (per station):** `PRCP`, `SNWD`, `TMAX`, `TMIN`, `TAVG`  
**Target:** Global `TAVG` (composite average temperature)  
**Dropped:** Geospatial columns (`LATITUDE`, `LONGITUDE`, `ELEVATION`) and `DATE` (no temporal trend detected)

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Visualised feature distributions using **histograms with KDE** (Seaborn) for temperature and precipitation across all three stations
- Generated **box plots** to identify spread and skewness per feature
- Plotted TAVG over time — confirmed no exploitable seasonal trend in the available data window

### 2. Data Preprocessing

| Step | Technique |
|------|-----------|
| Missing value analysis | `isnull().sum()` — up to 69% missingness in `PRCP_A` |
| Imputation | `sklearn.SimpleImputer` with mean strategy via `ColumnTransformer` pipeline |
| Outlier removal | IQR method (1.5× fence) applied across all numeric columns |
| Feature pruning | Dropped zero-importance features post Random Forest training |

### 3. Model Benchmarking

Three regressors were evaluated on an 80/20 train-test split (`random_state=42`):

| Model | MSE | R² Score |
|-------|-----|----------|
| Linear Regression | 43.86 | -3.50 |
| Huber Regressor | 31.25 | -2.21 |
| **Random Forest Regressor** | **1.97** | **0.798** |

> Linear and Huber regressors underperformed due to non-linear feature interactions between stations. Random Forest captured these naturally.

### 4. Feature Importance (Random Forest)

Top predictors ranked by impurity-based importance:

```
TAVG_B   → 14.1%
TMAX_C   → 13.8%
TMAX_B   → 13.0%
TAVG_C   → 12.2%
TMIN_B   → 11.9%
TMIN_A   → 11.1%
TMIN_C   → 10.0%
TMAX_A   →  8.9%
TAVG_A   →  5.0%
PRCP/SNWD → ~0% (removed)
```

Precipitation and snow depth features contributed negligible predictive signal and were dropped in the reduced model.

### 5. Hyperparameter Tuning

Exhaustive **GridSearchCV** (5-fold CV, `neg_mean_squared_error`) over:

```python
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4]
}
```

**Best Parameters:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`  
**Best Model MSE:** 1.97 | **R²:** 0.798

---

## 🧰 Tech Stack

```
Python 3.10          pandas · numpy · matplotlib · seaborn
scikit-learn         RandomForestRegressor · LinearRegression · HuberRegressor
                     SimpleImputer · ColumnTransformer · GridSearchCV
scipy / statsmodels  Statistical analysis support
Kaggle Notebooks     Execution environment
```

---

## 📁 Repository Structure

```
├── notebook.ipynb          # Full pipeline: EDA → preprocessing → modelling → submission
├── submission.csv          # Final predictions on test set
└── README.md               # This file
```

---

## 🚀 How to Reproduce

1. **Clone / open** the notebook in a Kaggle environment with the `shaastra-techathon-ai-ml-challenge-2` dataset attached.

2. **Run all cells** sequentially. Key pipeline stages:
   ```
   Cell 1–2   → Data loading
   Cell 3–17  → EDA and feature engineering
   Cell 21–32 → Missing value handling and outlier removal
   Cell 37–46 → Model training, evaluation, and tuning
   Cell 47–55 → Test set inference and submission export
   ```

3. The final `submission.csv` is written to `/kaggle/working/submission.csv`.

---

## 📊 Results

The tuned **Random Forest Regressor** achieved an **R² of 0.798** on the hold-out test split, explaining ~80% of variance in composite average temperature. Residuals followed a near-normal distribution, indicating well-calibrated predictions with no systematic bias.

---

## 💡 Key Learnings & Future Improvements

- **Temporal features** (month, season encoding) were dropped due to no visible trend but could improve generalisation with a larger dataset
- **Station-level averaging** of TAVG features dominated importance — a weighted ensemble of per-station sub-models could be explored
- **Gradient Boosting (XGBoost / LightGBM)** would likely outperform Random Forest with proper hyperparameter tuning given the tabular structure
- **KNN or iterative imputation** instead of mean imputation may better preserve covariance between station readings

---

## 👤 Author

**Debaditya** 

---

*Submitted as part of Shaastra Techathon — AI/ML Challenge Track, IIT Madras*
