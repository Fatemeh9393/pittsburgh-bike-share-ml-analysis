# Bike Share Rider Type Prediction
## POGOH Pittsburgh — Data Science Capstone Project

**Student:** Fatemeh Rahimeh Rahmati Kamel  
**University:** University of Pittsburgh — School of Computing and Information  
**Program:** Master of Science in Data Science  
**Date:** April 2026

---

## Project Overview

This project analyzes four years of bike-share trip data (2022–2025) from POGOH Pittsburgh to predict whether a rider is an annual **MEMBER** or a **CASUAL** user. The goal is to support POGOH's marketing and operations teams by enabling accurate rider segmentation for targeted membership conversion campaigns.

### Business Question
Can trip characteristics predict rider type (MEMBER vs. CASUAL), and what strategic value can these predictions provide to POGOH?

### Final Results
| Model | Accuracy | F1-Score (MEMBER) | ROC-AUC |
|-------|----------|-------------------|---------|
| Majority Baseline | 91.23% | 95.41% | N/A |
| Logistic Regression | 99.95% | 99.97% | N/A |
| **Random Forest (Final)** | **99.97%** | **99.99%** | **0.9999** |
| HistGradientBoosting | 99.98% | 99.99% | N/A |

**Best Model:** Random Forest  
**Key Finding:** Subscription type (product_category) is the dominant predictor (65% feature importance). Single Ride buyers are almost always Casual; Annual subscribers are almost always Members.

---

## Repository Structure

```
project/
│
├── Notebooks/
│   └── Assignment2.ipynb        # Main analysis notebook (full pipeline)
│
├── data/
│   ├── raw/                     # Original data files from POGOH
│   │   ├── 2022c.csv
│   │   ├── 2023c.csv
│   │   ├── 2024c.csv
│   │   ├── 2025c.csv
│   │   ├── STATION_EXPORT-1222.csv
│   │   ├── changed-capacities.csv
│   │   └── old-stations.csv
│   │
│   └── processed/               # Cleaned and engineered datasets
│       ├── train_data_final.csv     # Training set (2022–2024)
│       ├── val_data_final.csv       # Validation set (2024)
│       ├── test_data_final.csv      # Test set (2025)
│       └── feature_lists_final.json # Feature column lists
│
├── models/
│   ├── random_forest_final.pkl      # Final trained Random Forest model
│   └── hist_gb_final.pkl            # HistGradientBoosting model
│
├── Outputs/
│   ├── figures/                 # All visualizations
│   │   ├── confusion_matrix_final.png
│   │   ├── roc_curve_final.png
│   │   ├── feature_importance_final.png
│   │   ├── train_test_distribution.png
│   │   ├── log_transform.png
│   │   └── missing_data.png
│   │
│   ├── metrics/
│   │   ├── metrics.json             # All model performance metrics
│   │   └── feature_importance.csv   # Feature importance scores
│   │
│   └── results/
│       └── model_comparison.csv     # Model comparison summary
│
├── README.md                    # This file
└── Requirements.txt             # Python dependencies
```

---

## How to Reproduce the Analysis

### Step 1: Install Dependencies
```bash
pip install -r Requirements.txt
```

### Step 2: Place Raw Data
Make sure all raw CSV files are in `data/raw/`:
- 2022c.csv, 2023c.csv, 2024c.csv, 2025c.csv
- STATION_EXPORT-1222.csv

### Step 3: Run the Notebook
Open and run `Notebooks/Assignment2.ipynb` from top to bottom.

The notebook will automatically:
1. Load and combine 4 years of trip data
2. Clean and filter records (duration 1–240 min)
3. Engineer 12+ features (temporal, behavioral, station-based)
4. Split data by time (Train: 2022–2024, Test: 2025)
5. Train and compare 3 models
6. Save results, metrics, and figures to Outputs/

### Step 4: View Results
- Model metrics: `Outputs/metrics/metrics.json`
- Visualizations: `Outputs/figures/`
- Trained model: `models/random_forest_final.pkl`

---

## Key Analytical Decisions

| Decision | Rationale |
|----------|-----------|
| Time-based train/test split | Prevents data leakage; simulates real deployment |
| Duration filter: 1–240 min | Removes station tests and extreme outliers |
| Duration cap at 120 min | Domain-based outlier handling |
| class_weight='balanced' | Handles class imbalance (91% MEMBER / 9% CASUAL) |
| F1-score as primary metric | Accuracy misleading with imbalanced classes |
| Station popularity on train only | Prevents feature leakage to test set |
| Removed Product Name (raw) | Near-perfect correlation with target = leakage risk |

---

## Engineered Features

| Feature | Type | Description |
|---------|------|-------------|
| hour | Temporal | Hour of trip start |
| day_of_week | Temporal | Day name (Monday–Sunday) |
| month | Temporal | Month number |
| season | Temporal | Winter/Spring/Summer/Fall |
| time_of_day | Temporal | Morning/Afternoon/Evening/Night |
| is_weekend | Binary | 1 if Saturday or Sunday |
| is_rush_hour | Binary | 1 if 7–9 AM or 4–6 PM |
| is_round_trip | Binary | 1 if start = end station |
| is_e_assist | Binary | 1 if E-ASSIST bike model |
| duration_capped | Numeric | Duration capped at 120 min |
| duration_log | Numeric | Log-transformed duration |
| duration_category | Categorical | 5 bins: 0–10, 10–20, 20–30, 30–60, 60–120 min |
| start_station_popularity | Numeric | % of trips from this station (train only) |
| is_top_start_station | Binary | 1 if in top 10 stations by trip volume |
| product_category | Categorical | Single / Annual / Flex / Other |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

See `Requirements.txt` for full version details.

---

## Important Note on Model Performance

The Random Forest model achieves near-perfect accuracy (99.97%) primarily because the **product_category** feature encodes subscription type, which structurally determines rider type. This is documented transparently in the analysis. A future improvement would be training a second model without this feature to evaluate purely behavioral predictors.
