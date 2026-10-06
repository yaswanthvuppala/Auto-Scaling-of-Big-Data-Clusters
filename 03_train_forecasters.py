"""
03_train_forecasters.py  –  Train all four forecasting models on each dataset.

Pipeline for each dataset:
  1. Load processed time series (8 640 rows = 30 days × 288 steps/day).
  2. Chronological split: 60 % train / 20 % val / 20 % test.
  3. Train four models on train+val (val used for hyper-parameter tuning).
  4. Walk-forward predict on the TEST set (no data leakage).
  5. Save models (joblib) and predictions (CSV).

Walk-forward forecasting explained:
  At each test step t, the model sees actual values up to t−1 and predicts
  value at t.  Then we advance to t+1 and repeat.  This mirrors how an
  auto-scaler would operate in production: it always knows current load
  (from monitoring) and must predict the next interval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import config as cfg
from src.utils import load_processed, time_split
from src.forecasters import (
    MovingAverageForecaster,
    LinearRegressionForecaster,
    HoltWintersForecaster,
    GradientBoostingForecaster,
)


def train_and_predict(dataset_name: str, df: pd.DataFrame):
    """Train all four models on one dataset; return test predictions."""
    print(f"\n{'-'*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'-'*50}")

    # ── split ──────────────────────────────────────────────────
    train_df, val_df, test_df = time_split(df)

    train_cpu = train_df["cpu_utilization"].values
    val_cpu   = val_df["cpu_utilization"].values
    test_cpu  = test_df["cpu_utilization"].values

    # full series (train+val+test) for walk-forward
    full_cpu  = df["cpu_utilization"].values
    full_ts   = df["timestamp"].values

    test_start = len(train_df) + len(val_df)  # index where test begins

    results = {}

    # ── 1. Moving Average ──────────────────────────────────────
    print("\n  [1/4] Moving Average")
    ma = MovingAverageForecaster()
    ma.fit(train_cpu, val_cpu)
    ma_preds = ma.walk_forward(full_cpu, test_start)
    ma.save(cfg.MODELS_DIR / f"{dataset_name}_MA.pkl")
    results["MA"] = ma_preds

    # ── 2. Linear Regression ───────────────────────────────────
    print("\n  [2/4] Linear Regression")
    lr = LinearRegressionForecaster()
    lr.fit(train_cpu, val_cpu, timestamps=full_ts[:len(train_cpu)+len(val_cpu)])
    lr_preds = lr.walk_forward(full_cpu, test_start, timestamps=full_ts)
    lr.save(cfg.MODELS_DIR / f"{dataset_name}_LR.pkl")
    results["LR"] = lr_preds

    # ── 3. Holt-Winters ───────────────────────────────────────
    print("\n  [3/4] Holt-Winters")
    hw = HoltWintersForecaster()
    # Train on train+val combined (val already used for window selection above)
    train_val_cpu = np.concatenate([train_cpu, val_cpu])
    hw.fit(train_val_cpu, val_cpu)
    hw_preds = hw.walk_forward(full_cpu, test_start)
    hw.save(cfg.MODELS_DIR / f"{dataset_name}_HW.pkl")
    results["HW"] = hw_preds

    # ── 4. Gradient Boosting ──────────────────────────────────
    print("\n  [4/4] Gradient Boosting")
    gb = GradientBoostingForecaster()
    gb.fit(train_cpu, val_cpu, timestamps=full_ts[:len(train_cpu)+len(val_cpu)])
    gb_preds = gb.walk_forward(full_cpu, test_start, timestamps=full_ts)
    gb.save(cfg.MODELS_DIR / f"{dataset_name}_GB.pkl")
    results["GB"] = gb_preds

    # ── save predictions ──────────────────────────────────────
    pred_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "actual":    test_cpu,
    })
    for model_name, preds in results.items():
        pred_df[f"pred_{model_name}"] = preds

    # Also save total_cpu_demand for the simulator
    if "total_cpu_demand" in test_df.columns:
        pred_df["total_cpu_demand"] = test_df["total_cpu_demand"].values

    out_path = cfg.PREDICTIONS_DIR / f"{dataset_name}_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\n  Predictions saved -> {out_path.name}")

    return pred_df


# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("STEP 3: Train forecasters and generate test predictions")
    print("="*60)

    all_predictions = {}
    for name in cfg.DATASETS:
        df = load_processed(name)
        pred_df = train_and_predict(name, df)
        all_predictions[name] = pred_df

    print("\n[OK] All models trained and predictions saved in results/predictions/")
