from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataset_preview(df: pd.DataFrame, out_path: str | Path, n: int = 25) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(n).to_csv(out_path, index=False)
    return out_path


def plot_sales_trend(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = df.groupby("date", as_index=False)["qty_sold"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(g["date"], g["qty_sold"], linewidth=1.6)
    plt.title("Total Sales Trend (All Stores × Items)")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_category_sales(df: pd.DataFrame, out_path: str | Path, top_k: int = 10) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "category" not in df.columns:
        return out_path

    g = df.groupby("category", as_index=False)["qty_sold"].sum().sort_values("qty_sold", ascending=False).head(top_k)
    plt.figure(figsize=(10, 4))
    sns.barplot(data=g, x="category", y="qty_sold")
    plt.title(f"Top {top_k} Categories by Total Units Sold")
    plt.xlabel("Category")
    plt.ylabel("Units sold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_actual_vs_pred(sample_df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(sample_df["date"], sample_df["actual"], label="Actual", linewidth=1.6)
    plt.plot(sample_df["date"], sample_df["pred"], label="Predicted", linewidth=1.6)
    plt.title("Actual vs Predicted (Sample SKU-Store)")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(history_df["date"], history_df["qty_sold"], label="History", linewidth=1.6)
    plt.plot(forecast_df["date"], forecast_df["yhat"], label="Forecast", linewidth=1.8)
    plt.title("Forecast (Next Horizon)")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path
