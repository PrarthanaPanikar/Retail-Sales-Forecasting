from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"store_id", "item_id", "date", "qty_sold"}


@dataclass(frozen=True)
class IngestionReport:
    n_rows: int
    n_duplicates: int
    n_missing_qty: int
    n_negative_qty: int
    n_stockout_rows: int


def load_sales_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def basic_quality_checks(df: pd.DataFrame) -> IngestionReport:
    n_duplicates = int(df.duplicated(["store_id", "item_id", "date"]).sum())
    n_missing_qty = int(df["qty_sold"].isna().sum())
    n_negative_qty = int((df["qty_sold"] < 0).sum())
    n_stockout_rows = int(df["stockout_flag"].sum()) if "stockout_flag" in df.columns else 0
    return IngestionReport(
        n_rows=int(len(df)),
        n_duplicates=n_duplicates,
        n_missing_qty=n_missing_qty,
        n_negative_qty=n_negative_qty,
        n_stockout_rows=n_stockout_rows,
    )


def clean_sales_data(df: pd.DataFrame, drop_stockout_censored: bool = True) -> pd.DataFrame:
    """
    Minimal, industry-style cleaning:
    - ensures correct dtypes
    - removes duplicates
    - handles missing promo/price fields
    - optionally removes stockout-censored rows (stockout_flag==1)
    """
    df = df.copy()

    # Drop duplicates on grain
    df = df.drop_duplicates(["store_id", "item_id", "date"], keep="last")

    # Enforce non-negative
    df["qty_sold"] = pd.to_numeric(df["qty_sold"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    for c, default in [("on_promo", 0), ("discount_pct", 0.0), ("price", 0.0)]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)

    if drop_stockout_censored and "stockout_flag" in df.columns:
        df = df[df["stockout_flag"] == 0].copy()

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
