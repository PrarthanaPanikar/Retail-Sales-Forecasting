from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    n_stores: int = 5
    n_items: int = 80
    n_categories: int = 8
    seed: int = 42


def _date_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def simulate_retail_timeseries(cfg: SimulationConfig) -> pd.DataFrame:
    """
    Create a realistic beginner-friendly synthetic dataset for retail demand forecasting.

    Output grain: store_id × item_id × date (daily)

    Columns include:
    - qty_sold (target)
    - price, discount_pct, on_promo
    - stock_on_hand, stockout_flag (censoring indicator)
    - product/store metadata (category, brand, city_cluster)
    - supply assumptions (lead_time_days, unit_cost, ordering_cost, holding_cost_rate)
    """
    rng = np.random.default_rng(cfg.seed)
    dates = _date_range(cfg.start_date, cfg.end_date)
    n_days = len(dates)

    stores = pd.DataFrame(
        {
            "store_id": [f"S{idx:02d}" for idx in range(1, cfg.n_stores + 1)],
            "city_cluster": rng.choice(["metro", "tier_1", "tier_2"], size=cfg.n_stores, replace=True),
            "footfall_index": rng.normal(loc=1.0, scale=0.15, size=cfg.n_stores).clip(0.6, 1.5),
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [f"I{idx:04d}" for idx in range(1, cfg.n_items + 1)],
            "category": [f"C{c:02d}" for c in rng.integers(1, cfg.n_categories + 1, size=cfg.n_items)],
            "brand": rng.choice(["A", "B", "C", "D", "E"], size=cfg.n_items, replace=True),
            "pack_size": rng.choice([1, 2, 5, 10], size=cfg.n_items, replace=True),
            "base_price": rng.lognormal(mean=4.2, sigma=0.35, size=cfg.n_items).round(2),  # ~ 40-1200
            "unit_cost": rng.lognormal(mean=3.7, sigma=0.35, size=cfg.n_items).round(2),
            "holding_cost_rate": rng.uniform(0.18, 0.30, size=cfg.n_items).round(3),
            "ordering_cost": rng.uniform(200, 900, size=cfg.n_items).round(0),
            "lead_time_days": rng.integers(3, 15, size=cfg.n_items),
        }
    )

    # Make some items intermittent (lots of zeros)
    intermittent_flag = rng.random(cfg.n_items) < 0.25
    base_rate = rng.gamma(shape=2.0, scale=5.0, size=cfg.n_items)  # baseline daily demand
    base_rate[intermittent_flag] *= 0.25

    # Seasonality & events (simple but realistic)
    dow = pd.Series(dates).dt.dayofweek.values  # 0..6
    weekly = np.where(np.isin(dow, [5, 6]), 1.15, 0.95)  # weekends higher

    doy = pd.Series(dates).dt.dayofyear.values
    yearly = 1.0 + 0.15 * np.sin(2 * np.pi * doy / 365.25)  # smooth seasonality

    # Festival spikes (few days)
    festival_days = pd.to_datetime(
        [
            "2024-10-30",
            "2024-11-01",
            "2025-10-20",
            "2025-10-22",
            "2025-12-25",
        ]
    )
    festival_flag = pd.Series(dates).isin(festival_days).astype(int).values
    festival = 1.0 + 0.35 * festival_flag

    calendar = pd.DataFrame(
        {
            "date": dates,
            "dow": dow,
            "weekofyear": pd.Series(dates).dt.isocalendar().week.astype(int).values,
            "month": pd.Series(dates).dt.month.astype(int).values,
            "festival_flag": festival_flag,
        }
    )

    rows: list[pd.DataFrame] = []
    for _, s in stores.iterrows():
        store_mult = float(s["footfall_index"]) * (1.08 if s["city_cluster"] == "metro" else 1.0)

        # store-level promo intensity
        promo_base = rng.uniform(0.07, 0.14)
        promo_prob = (promo_base * weekly).clip(0.03, 0.25)

        # Promotions per day for this store (applied item-wise with item sensitivity)
        on_promo_day = rng.random(n_days) < promo_prob
        discount_day = np.where(on_promo_day, rng.uniform(0.05, 0.35, size=n_days), 0.0)

        for idx_item, it in items.iterrows():
            item_mult = base_rate[idx_item]

            # price variability
            base_price = float(it["base_price"])
            price_noise = rng.normal(loc=0.0, scale=0.03, size=n_days)
            price = (base_price * (1 + price_noise)).clip(1.0, None)

            # promo sensitivity (lift)
            promo_sens = rng.uniform(0.8, 2.0)
            demand_mult = weekly * yearly * festival
            promo_lift = 1.0 + promo_sens * discount_day

            # expected demand (lambda) and realized demand
            lam = (store_mult * item_mult * demand_mult * promo_lift).clip(0.0, None)

            # Add intermittency via zero-inflation
            p_zero = 0.55 if intermittent_flag[idx_item] else 0.08
            zeros = rng.random(n_days) < p_zero
            qty = rng.poisson(lam).astype(int)
            qty[zeros] = 0

            # Inventory simulation: on-hand fluctuates; stockouts censor sales
            lead_time = int(it["lead_time_days"])
            init_on_hand = int(max(10, rng.normal(loc=120, scale=40)))
            on_hand = np.zeros(n_days, dtype=int)
            stockout = np.zeros(n_days, dtype=int)
            sold = np.zeros(n_days, dtype=int)

            # Simple replenishment: reorder when low (not optimized; used to create stockouts)
            reorder_point = int(max(20, np.quantile(qty, 0.80) * lead_time))
            order_up_to = int(reorder_point * 2.2)

            pipeline = []  # (arrival_day_index, qty)
            current = init_on_hand

            for t in range(n_days):
                # arrivals
                if pipeline:
                    arrivals_today = sum(q for day, q in pipeline if day == t)
                    if arrivals_today > 0:
                        current += arrivals_today
                    pipeline = [(day, q) for day, q in pipeline if day != t]

                # demand vs available stock
                demand_t = int(qty[t])
                if demand_t > current:
                    sold[t] = current
                    stockout[t] = 1
                    current = 0
                else:
                    sold[t] = demand_t
                    current -= demand_t

                # reorder decision (rule-based)
                if current <= reorder_point:
                    order_qty = max(0, order_up_to - current)
                    if order_qty > 0:
                        pipeline.append((t + lead_time, int(order_qty)))

                on_hand[t] = current

            df_si = pd.DataFrame(
                {
                    "store_id": s["store_id"],
                    "item_id": it["item_id"],
                    "date": dates,
                    "qty_sold": sold,
                    "price": price.round(2),
                    "discount_pct": discount_day.round(3),
                    "on_promo": on_promo_day.astype(int),
                    "stock_on_hand": on_hand,
                    "stockout_flag": stockout,
                    "category": it["category"],
                    "brand": it["brand"],
                    "city_cluster": s["city_cluster"],
                    "footfall_index": float(s["footfall_index"]),
                    "lead_time_days": lead_time,
                    "unit_cost": float(it["unit_cost"]),
                    "ordering_cost": float(it["ordering_cost"]),
                    "holding_cost_rate": float(it["holding_cost_rate"]),
                }
            )
            rows.append(df_si)

    df = pd.concat(rows, ignore_index=True)
    df = df.merge(calendar, on="date", how="left")
    return df


def save_simulated_dataset(
    out_path: str | Path,
    cfg: SimulationConfig = SimulationConfig(),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = simulate_retail_timeseries(cfg)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = save_simulated_dataset(Path("data/raw/retail_timeseries.csv"))
    print(f"Saved: {path}")
