from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (1, 7, 14, 28)
    windows: tuple[int, ...] = (7, 14, 28)
    add_price_promo_features: bool = True


TARGET_COL = "qty_sold"
DATE_COL = "date"


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df[DATE_COL]
    df["dow"] = d.dt.dayofweek.astype(int)
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month.astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    return df


def build_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """
    Builds time-series features per store_id × item_id:
    - lag features for demand
    - rolling mean/std (shifted to avoid leakage)
    - calendar features
    - optional price/promo interactions
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(["store_id", "item_id", DATE_COL])

    def featurize_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(DATE_COL).copy()
        for L in cfg.lags:
            g[f"lag_{L}"] = g[TARGET_COL].shift(L)
        for W in cfg.windows:
            shifted = g[TARGET_COL].shift(1)
            g[f"roll_mean_{W}"] = shifted.rolling(W).mean()
            g[f"roll_std_{W}"] = shifted.rolling(W).std()
            g[f"roll_min_{W}"] = shifted.rolling(W).min()
            g[f"roll_max_{W}"] = shifted.rolling(W).max()
        return g

    # Some pandas versions exclude grouping columns from `groupby().apply(...)`.
    # We restore them from the MultiIndex if needed.
    df = df.groupby(["store_id", "item_id"], group_keys=True).apply(featurize_group)
    if "store_id" not in df.columns or "item_id" not in df.columns:
        df = df.reset_index(level=[0, 1]).rename(columns={"level_0": "store_id", "level_1": "item_id"})
    df = df.reset_index(drop=True)
    df = _add_calendar_features(df)

    if cfg.add_price_promo_features:
        if "discount_pct" in df.columns:
            df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0.0)
        else:
            df["discount_pct"] = 0.0
        if "on_promo" in df.columns:
            df["on_promo"] = pd.to_numeric(df["on_promo"], errors="coerce").fillna(0).astype(int)
        else:
            df["on_promo"] = 0
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        else:
            df["price"] = 0.0

        df["promo_discount"] = df["on_promo"] * df["discount_pct"]
        df["price_after_discount"] = (df["price"] * (1 - df["discount_pct"])).clip(lower=0.0)

    # Clean up infinite/NaN from rolling std early rows
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def finalize_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops cold-start NaNs created by lags/rolling features.
    Keeps only rows where features are fully defined.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL}]
    df = df.dropna(subset=feature_cols)
    return df.reset_index(drop=True)
