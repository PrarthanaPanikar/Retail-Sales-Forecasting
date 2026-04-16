from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype


TARGET_COL = "qty_sold"
DATE_COL = "date"


@dataclass(frozen=True)
class TrainConfig:
    test_days: int = 56  # last 8 weeks held out
    random_state: int = 13
    n_estimators: int = 60
    max_depth: int | None = 14
    min_samples_leaf: int = 2
    n_jobs: int = -1


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics_path: Path
    metrics: dict
    feature_cols: list[str]


def _infer_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {TARGET_COL, DATE_COL}
    return [c for c in df.columns if c not in drop]


def temporal_train_test_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(DATE_COL).copy()
    cutoff = df[DATE_COL].max() - pd.Timedelta(days=test_days - 1)
    train_df = df[df[DATE_COL] < cutoff].copy()
    test_df = df[df[DATE_COL] >= cutoff].copy()
    return train_df, test_df


def fit_random_forest(train_df: pd.DataFrame, cfg: TrainConfig) -> RandomForestRegressor:
    feature_cols = _infer_feature_cols(train_df)
    X = train_df[feature_cols].copy()
    y = train_df[TARGET_COL]

    obj_cols = [c for c in feature_cols if not is_numeric_dtype(X[c])]
    num_cols = [c for c in feature_cols if c not in obj_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), obj_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    model = Pipeline([("pre", pre), ("rf", rf)])
    model.fit(X, y)
    return model  # type: ignore[return-value]


def evaluate(model, test_df: pd.DataFrame) -> dict:
    feature_cols = _infer_feature_cols(test_df)
    X = test_df[feature_cols].copy()
    y_true = test_df[TARGET_COL].to_numpy()
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Seasonal naive baseline: last week's same DOW (lag_7 if present)
    if "lag_7" in test_df.columns:
        naive = test_df["lag_7"].to_numpy()
        naive_mae = float(mean_absolute_error(y_true, naive))
        mase = float(mae / naive_mae) if naive_mae > 1e-9 else float("nan")
    else:
        naive_mae = float("nan")
        mase = float("nan")

    return {"mae": mae, "rmse": rmse, "naive_mae": naive_mae, "mase": mase}


def train_and_save(
    df: pd.DataFrame,
    out_dir: str | Path = "models",
    cfg: TrainConfig = TrainConfig(),
) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = temporal_train_test_split(df, test_days=cfg.test_days)
    model = fit_random_forest(train_df, cfg)
    metrics = evaluate(model, test_df)
    feature_cols = _infer_feature_cols(df)

    model_path = out_dir / "rf_demand_forecaster.joblib"
    metrics_path = out_dir / "metrics.json"

    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2))

    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        metrics=metrics,
        feature_cols=feature_cols,
    )


def load_model(path: str | Path):
    artifact = joblib.load(Path(path))
    return artifact["model"], artifact["feature_cols"]
