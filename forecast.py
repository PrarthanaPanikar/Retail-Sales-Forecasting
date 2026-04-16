from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COL = "qty_sold"
DATE_COL = "date"


@dataclass(frozen=True)
class ForecastConfig:
    horizon_days: int = 28


def _next_date(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.Timedelta(days=1)


def _calendar_row(date: pd.Timestamp) -> dict:
    return {
        "date": date,
        "dow": int(date.dayofweek),
        "weekofyear": int(date.isocalendar().week),
        "month": int(date.month),
        "is_weekend": int(date.dayofweek in (5, 6)),
    }


def _compute_lag_features(history: pd.Series, lags: list[int]) -> dict:
    feats: dict[str, float] = {}
    for L in lags:
        feats[f"lag_{L}"] = float(history.iloc[-L]) if len(history) >= L else np.nan
    return feats


def _compute_roll_features(history: pd.Series, windows: list[int]) -> dict:
    feats: dict[str, float] = {}
    shifted = history.iloc[:-1]  # shift(1) effect: use data up to t-1 for day t
    for W in windows:
        tail = shifted.iloc[-W:] if len(shifted) >= W else shifted
        feats[f"roll_mean_{W}"] = float(tail.mean()) if len(tail) else np.nan
        feats[f"roll_std_{W}"] = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
        feats[f"roll_min_{W}"] = float(tail.min()) if len(tail) else np.nan
        feats[f"roll_max_{W}"] = float(tail.max()) if len(tail) else np.nan
    return feats


def _default_exogenous(last_row: pd.Series) -> dict:
    """
    For a student portfolio, we keep future exogenous variables simple:
    - assume future price equals latest price
    - assume promo off (0) unless you later plug in promo calendars
    """
    return {
        "price": float(last_row["price"]) if "price" in last_row else 0.0,
        "discount_pct": 0.0,
        "on_promo": 0,
        "promo_discount": 0.0,
        "price_after_discount": float(last_row["price"]) if "price" in last_row else 0.0,
    }


def recursive_forecast_for_series(
    model,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: ForecastConfig = ForecastConfig(),
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Produce a daily forecast for one store_id×item_id using recursive prediction.

    Requirements:
    - history_df must include at least: date, qty_sold
    - should include exogenous columns if your model uses them (price/on_promo/discount_pct)
    """
    if lags is None:
        lags = [1, 7, 14, 28]
    if windows is None:
        windows = [7, 14, 28]

    hist = history_df.sort_values(DATE_COL).copy()
    if len(hist) < max(lags) + 2:
        raise ValueError("Not enough history for forecasting. Generate more history or reduce lags.")

    # We'll build features for each future day using the evolving demand history
    demand_hist = hist[TARGET_COL].astype(float).reset_index(drop=True)
    last_date = pd.to_datetime(hist[DATE_COL].max())
    last_row = hist.iloc[-1]

    preds: list[dict] = []
    for step in range(1, cfg.horizon_days + 1):
        date = last_date + pd.Timedelta(days=step)

        feats: dict[str, float] = {}
        feats.update(_calendar_row(date))
        feats.update(_compute_lag_features(demand_hist, lags))
        feats.update(_compute_roll_features(demand_hist, windows))
        feats.update(_default_exogenous(last_row))

        # carry stable identifiers / metadata if present in model features
        for key in ["store_id", "item_id", "category", "brand", "city_cluster", "lead_time_days", "unit_cost", "ordering_cost", "holding_cost_rate", "footfall_index", "festival_flag"]:
            if key in hist.columns and key in feature_cols:
                feats[key] = last_row[key]

        X = pd.DataFrame([feats])
        # ensure all features exist
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]
        X = X.fillna(0.0)

        yhat = float(model.predict(X)[0])
        yhat = max(0.0, yhat)

        preds.append({"date": date, "yhat": yhat})
        demand_hist = pd.concat([demand_hist, pd.Series([yhat])], ignore_index=True)

    return pd.DataFrame(preds)
