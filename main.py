from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest import basic_quality_checks, clean_sales_data, load_sales_csv, save_processed
from src.data.simulate_retail_data import SimulationConfig, save_simulated_dataset
from src.features.build_features import FeatureConfig, build_features, finalize_model_frame
from src.inventory.policy import InventoryInputs, annualize_demand, compute_inventory_decision
from src.models.forecast import ForecastConfig, recursive_forecast_for_series
from src.models.train import TrainConfig, load_model, train_and_save, temporal_train_test_split
from src.viz.reporting import (
    plot_actual_vs_pred,
    plot_category_sales,
    plot_forecast,
    plot_sales_trend,
    save_dataset_preview,
)


RAW_PATH = Path("data/raw/retail_timeseries.csv")
PROCESSED_PATH = Path("data/processed/retail_timeseries_clean.csv")
MAX_GROUPS_FOR_TRAIN = 80  # speed-friendly for student laptops


def _maybe_generate_data() -> None:
    if RAW_PATH.exists():
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = SimulationConfig(
        start_date="2024-01-01",
        end_date="2025-12-31",
        n_stores=5,
        n_items=80,
        n_categories=8,
        seed=42,
    )
    save_simulated_dataset(RAW_PATH, cfg=cfg)


def _build_recommendations(
    df_hist: pd.DataFrame,
    model,
    feature_cols: list[str],
    horizon_days: int = 28,
) -> pd.DataFrame:
    """
    Generate replenishment recommendations for each store_id×item_id using:
    - ML forecast for next horizon
    - SS/ROP based on service level and residual uncertainty
    - EOQ based on annualized demand + holding/ordering costs
    """
    # Residual std proxy (global) from a simple holdout evaluation
    feat_df = finalize_model_frame(build_features(df_hist))
    train_df, test_df = temporal_train_test_split(feat_df, test_days=56)
    y_true = test_df["qty_sold"].to_numpy()
    y_pred = np.clip(model.predict(test_df[feature_cols]), 0, None)
    resid_std = float(np.std(y_true - y_pred))

    # For a fast, beginner-friendly batch job, we compute replenishment using
    # a simple demand forecast proxy (recent mean). The Streamlit app and sample
    # charts use the ML model forecast.
    series_sizes = df_hist.groupby(["store_id", "item_id"]).size().sort_values(ascending=False)
    top_keys = list(series_sizes.head(120).index)

    recs: list[dict] = []
    for (store_id, item_id) in top_keys:
        g = df_hist[(df_hist["store_id"] == store_id) & (df_hist["item_id"] == item_id)].sort_values("date").copy()
        g = g.sort_values("date").copy()
        last = g.iloc[-1]

        recent_mean = float(g["qty_sold"].tail(14).mean())
        recent_mean = max(0.0, recent_mean)
        fc_arr = np.repeat(recent_mean, horizon_days)

        lead_time = int(last["lead_time_days"]) if "lead_time_days" in last else 7
        unit_cost = float(last["unit_cost"]) if "unit_cost" in last else 100.0
        ordering_cost = float(last["ordering_cost"]) if "ordering_cost" in last else 500.0
        holding_rate = float(last["holding_cost_rate"]) if "holding_cost_rate" in last else 0.25
        on_hand = float(last["stock_on_hand"]) if "stock_on_hand" in last else 0.0

        daily_mean = float(g["qty_sold"].tail(56).mean()) if len(g) >= 56 else float(g["qty_sold"].mean())
        annual_demand = annualize_demand(daily_mean)

        inv_in = InventoryInputs(
            on_hand=on_hand,
            lead_time_days=lead_time,
            service_level=0.95,
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            unit_cost=unit_cost,
            holding_cost_rate=holding_rate,
        )
        decision = compute_inventory_decision(
            forecast=fc_arr,
            resid_std=resid_std,
            inv=inv_in,
        )

        recs.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "category": last.get("category", None),
                "lead_time_days": lead_time,
                "on_hand": on_hand,
                "service_level": inv_in.service_level,
                "forecast_next_%dd_sum" % lead_time: float(fc_arr[:lead_time].sum()),
                "safety_stock": decision.safety_stock,
                "reorder_point": decision.reorder_point,
                "eoq": decision.eoq,
                "recommended_order_qty": decision.order_qty,
            }
        )

    out = pd.DataFrame(recs)
    out = out.sort_values(["recommended_order_qty"], ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    _maybe_generate_data()

    df_raw = load_sales_csv(RAW_PATH)
    report = basic_quality_checks(df_raw)
    print("Ingestion checks:", report)

    df_clean = clean_sales_data(df_raw, drop_stockout_censored=True)
    save_processed(df_clean, PROCESSED_PATH)

    # Proof assets (EDA)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    save_dataset_preview(df_clean, "outputs/tables/dataset_preview.csv")
    plot_sales_trend(df_clean, "outputs/figures/sales_trend_total.png")
    plot_category_sales(df_clean, "outputs/figures/category_sales_top.png")

    # Features + Training (subset for speed)
    groups = (
        df_clean.groupby(["store_id", "item_id"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    if len(groups) > MAX_GROUPS_FOR_TRAIN:
        groups = groups.sample(n=MAX_GROUPS_FOR_TRAIN, random_state=13)
    df_train_base = df_clean.merge(groups[["store_id", "item_id"]], on=["store_id", "item_id"], how="inner")

    feat_cfg = FeatureConfig()
    df_feat = finalize_model_frame(build_features(df_train_base, feat_cfg))

    train_res = train_and_save(df_feat, out_dir="models", cfg=TrainConfig())
    print("Saved model:", train_res.model_path)
    print("Metrics:", train_res.metrics)

    model, feature_cols = load_model(train_res.model_path)

    # Actual vs predicted (sample SKU-store on holdout)
    train_df, test_df = temporal_train_test_split(df_feat, test_days=56)
    sample_key = test_df.groupby(["store_id", "item_id"])["qty_sold"].sum().sort_values(ascending=False).head(1).index[0]
    sample = test_df[(test_df["store_id"] == sample_key[0]) & (test_df["item_id"] == sample_key[1])].sort_values("date")
    pred = np.clip(model.predict(sample[feature_cols]), 0, None)
    avp = pd.DataFrame({"date": sample["date"].values, "actual": sample["qty_sold"].values, "pred": pred})
    plot_actual_vs_pred(avp, "outputs/figures/actual_vs_pred_sample.png")

    # Forecast + Inventory decision (same sample using full clean history)
    g_hist = df_clean[(df_clean["store_id"] == sample_key[0]) & (df_clean["item_id"] == sample_key[1])].sort_values("date")
    fc = recursive_forecast_for_series(model, g_hist, feature_cols, cfg=ForecastConfig(horizon_days=28))
    plot_forecast(g_hist.tail(120), fc, "outputs/figures/forecast_sample.png")

    # Recommendations for all items
    recs = _build_recommendations(df_clean, model, feature_cols, horizon_days=28)
    recs.to_csv("outputs/tables/reorder_recommendations.csv", index=False)
    print("Saved recommendations: outputs/tables/reorder_recommendations.csv")


if __name__ == "__main__":
    main()
