from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.ingest import clean_sales_data, load_sales_csv
from src.inventory.policy import InventoryInputs, annualize_demand, compute_inventory_decision
from src.models.forecast import ForecastConfig, recursive_forecast_for_series


st.set_page_config(page_title="Retail Forecast & Inventory Optimizer", layout="wide")
st.title("Retail Sales Forecasting & Inventory Optimization System")
st.caption("Student portfolio demo: forecast demand → compute Safety Stock, ROP, EOQ → recommend order quantity.")

DATA_PATH = Path("data/raw/retail_timeseries.csv")
MODEL_PATH = Path("models/rf_demand_forecaster.joblib")


@st.cache_data
def _load_data() -> pd.DataFrame:
    df = load_sales_csv(DATA_PATH)
    return clean_sales_data(df, drop_stockout_censored=True)


@st.cache_resource
def _load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["feature_cols"]


if not DATA_PATH.exists():
    st.error("Dataset not found. Run `python main.py` once to generate data and train the model.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("Model not found. Run `python main.py` once to train and save the model.")
    st.stop()

df = _load_data()
model, feature_cols = _load_model()

colA, colB, colC = st.columns(3)
store_id = colA.selectbox("Store", sorted(df["store_id"].unique()))
item_id = colB.selectbox("Item", sorted(df[df["store_id"] == store_id]["item_id"].unique()))
horizon_days = colC.slider("Forecast horizon (days)", min_value=7, max_value=56, value=28, step=7)

g = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date").copy()
last = g.iloc[-1]

left, right = st.columns([1, 1])
with left:
    st.subheader("Current item snapshot")
    st.write(
        {
            "category": last.get("category", None),
            "brand": last.get("brand", None),
            "lead_time_days": int(last.get("lead_time_days", 7)),
            "stock_on_hand": int(last.get("stock_on_hand", 0)),
            "unit_cost": float(last.get("unit_cost", 100.0)),
            "ordering_cost": float(last.get("ordering_cost", 500.0)),
            "holding_cost_rate": float(last.get("holding_cost_rate", 0.25)),
        }
    )

with right:
    st.subheader("Planner inputs (editable)")
    on_hand = st.number_input("On-hand units", min_value=0, value=int(last.get("stock_on_hand", 0)))
    lead_time = st.number_input("Lead time (days)", min_value=1, value=int(last.get("lead_time_days", 7)))
    service = st.selectbox("Target service level", options=[0.90, 0.95, 0.97, 0.99], index=1)

    unit_cost = st.number_input("Unit cost", min_value=0.0, value=float(last.get("unit_cost", 100.0)))
    ordering_cost = st.number_input("Ordering cost per PO", min_value=0.0, value=float(last.get("ordering_cost", 500.0)))
    holding_rate = st.number_input("Holding cost rate (per year)", min_value=0.0, max_value=1.0, value=float(last.get("holding_cost_rate", 0.25)))


st.divider()
st.subheader("Forecast + Replenishment Recommendation")

fc = recursive_forecast_for_series(
    model=model,
    history_df=g,
    feature_cols=feature_cols,
    cfg=ForecastConfig(horizon_days=int(horizon_days)),
)

# Residual std proxy: use recent demand volatility as a simple uncertainty estimate for the demo
resid_std = float(g["qty_sold"].tail(56).std(ddof=1)) if len(g) >= 56 else float(g["qty_sold"].std(ddof=1))
resid_std = 0.0 if np.isnan(resid_std) else resid_std

daily_mean = float(g["qty_sold"].tail(56).mean()) if len(g) >= 56 else float(g["qty_sold"].mean())
annual_demand = annualize_demand(daily_mean)

inv_in = InventoryInputs(
    on_hand=float(on_hand),
    lead_time_days=int(lead_time),
    service_level=float(service),
    annual_demand=float(annual_demand),
    ordering_cost=float(ordering_cost),
    unit_cost=float(unit_cost),
    holding_cost_rate=float(holding_rate),
)
decision = compute_inventory_decision(fc["yhat"].to_numpy(), resid_std=resid_std, inv=inv_in)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Safety Stock (SS)", f"{decision.safety_stock:.1f}")
metric2.metric("Reorder Point (ROP)", f"{decision.reorder_point:.1f}")
metric3.metric("EOQ", f"{decision.eoq:.1f}")
metric4.metric("Recommended Order Qty", f"{decision.order_qty:.1f}")

chart_df = pd.concat(
    [
        g[["date", "qty_sold"]].tail(120).rename(columns={"qty_sold": "value"}).assign(series="history"),
        fc.rename(columns={"yhat": "value"}).assign(series="forecast"),
    ],
    ignore_index=True,
)
st.line_chart(chart_df, x="date", y="value", color="series")

st.write("Forecast table (download for proof).")
st.dataframe(fc, use_container_width=True)
st.download_button(
    "Download forecast CSV",
    data=fc.to_csv(index=False).encode("utf-8"),
    file_name=f"forecast_{store_id}_{item_id}.csv",
    mime="text/csv",
)
