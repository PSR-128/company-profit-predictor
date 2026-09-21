# Company Profit Predictor 💰

A machine learning script that predicts a company's profit from its spending pattern, using scikit-learn's Linear Regression.

## Overview

`company_profit_prediction.py` loads `1000_Companies.csv` and trains a linear regression model to predict `Profit` from R&D Spend, Administration, Marketing Spend, and State.

### Dataset

`1000_Companies.csv` (included in this repo) contains, per company:

| Column | Description |
|---|---|
| `R&D Spend` | Amount spent on research & development |
| `Administration` | Administrative costs |
| `Marketing Spend` | Marketing expenditure |
| `State` | State the company operates in (categorical) |
| `Profit` | Target variable — the company's profit |

### Preprocessing & Model

1. `State` is label-encoded into numeric form with `LabelEncoder`.
2. Data is split 80/20 into train/test sets (`random_state=42`).
3. A `LinearRegression` model is fit on the training data to predict `Profit`.
4. Model performance is evaluated on the test set via R².

**Model accuracy (R²): 98.3%** — i.e., the model explains about 98.3% of the variance in company profit on the held-out test data.

> The script also has a commented-out Seaborn heatmap (`sns.heatmap(df.corr(), ...)`) for visualizing feature correlations — uncomment it if you want to explore the data visually.

## Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn seaborn matplotlib
```

### Run

```bash
python company_profit_prediction.py
```

This prints the R-squared value of the model on the test set.

## Project Structure

```
company-profit-predictor/
├── company_profit_prediction.py   # Data loading, preprocessing, training, and evaluation
└── 1000_Companies.csv             # Training dataset
```

## License

No license specified yet — add one (e.g. MIT) if you plan to share or accept contributions to this project.
