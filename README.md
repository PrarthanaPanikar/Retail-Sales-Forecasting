# Retail Sales Forecasting & Inventory Optimization System

Beginner-friendly, industry-aligned portfolio project that simulates a retail environment, forecasts demand at **store × item × day** level, and converts forecasts into replenishment decisions using **Safety Stock (SS)**, **Reorder Point (ROP)**, and **Economic Order Quantity (EOQ)**.

## Why this project matters (industry relevance)
Retailers (supermarkets, D2C, e-commerce marketplaces) need the right stock at the right time:
- **Stockouts** → lost sales + unhappy customers (lower fill rate)
- **Overstock** → cash stuck in inventory + holding/expiry risk (higher working capital)
- **Forecast-driven replenishment** improves service levels while reducing inventory cost.

## What you will build
- **Virtual retail dataset** (synthetic but realistic): stores, items, categories, promos, seasonality, stockouts
- **EDA + proof outputs**: trend charts, category charts, tables
- **Forecast model**: Random Forest on lag/rolling + calendar + promo/price features
- **Inventory optimizer**: service-level based SS/ROP + EOQ + order quantity recommendation
- **Dashboard (Streamlit)**: select SKU/store → see forecast & recommended order

## Architecture (high-level)
`data/raw` → ingestion/cleaning → feature engineering → model training → forecasting → inventory policy → outputs + dashboard

## Folder structure
```
Retail Sales Forecasting/
├── app/
│   └── app.py                       # Streamlit dashboard
├── data/
│   ├── raw/                         # generated dataset (CSV)
│   └── processed/                   # cleaned dataset
├── models/                          # saved ML model + metrics
├── outputs/
│   ├── figures/                     # charts (PNG)
│   └── tables/                      # CSV outputs
├── src/
│   ├── data/                        # simulation + ingestion
│   ├── features/                    # feature engineering
│   ├── inventory/                   # SS/ROP/EOQ
│   ├── models/                      # training + forecasting
│   └── viz/                         # plots/reporting
├── main.py                          # end-to-end pipeline runner
├── requirements.txt
└── .gitignore
```

## Setup
Recommended: **Python 3.10–3.12** (most data science libraries publish stable wheels for these versions).

### Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Mac/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (end-to-end)
Generate synthetic data → train model → export outputs:
```bash
python main.py
```

Outputs created:
- `data/raw/retail_timeseries.csv`
- `data/processed/retail_timeseries_clean.csv`
- `models/rf_demand_forecaster.joblib`
- `models/metrics.json`
- `outputs/figures/*.png`
- `outputs/tables/reorder_recommendations.csv`

## Run the dashboard
```bash
streamlit run app/app.py
```

## Proof assets (what to screenshot)
- `outputs/tables/dataset_preview.csv` opened in Excel/VSCode
- `outputs/figures/sales_trend_total.png`
- `outputs/figures/category_sales_top.png`
- `outputs/figures/actual_vs_pred_sample.png`
- `outputs/figures/forecast_sample.png`
- `outputs/tables/reorder_recommendations.csv`
- Streamlit page with metrics + chart

## Future upgrades
- Per-SKU model selection (Croston/SBA for intermittent demand)
- Proper rolling-origin backtesting per series
- Promo calendar planning + price elasticity modeling
- Multi-echelon inventory (DC → stores)
- Monitoring/drift checks + scheduled retraining

## Author
Your Name  
LinkedIn: <link> • GitHub: <link>
