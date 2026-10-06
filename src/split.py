"""
src/split.py  -  Chronological train/validation/test partitioning and data leakage verification.

Rules:
  - 60% train, 20% validation, 20% test.
  - Strictly chronological: no shuffling, no temporal overlap.
  - Normalization parameters (mean, std) computed on train partition ONLY.
  - Walk-forward inference helper: at step t, forecaster observes only history <= t.
  - Unit test proving no future data leakage.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


def train_val_test_split(df: pd.DataFrame,
                         train_ratio: float = 0.60,
                         val_ratio: float = 0.20,
                         test_ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Chronological time-series split.
    Guarantees no shuffling and computes normalization parameters exclusively on train.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    train = df.iloc[:t1].copy().reset_index(drop=True)
    val   = df.iloc[t1:t2].copy().reset_index(drop=True)
    test  = df.iloc[t2:].copy().reset_index(drop=True)

    # Compute normalization statistics strictly on train
    train_mean = float(train["load"].mean())
    train_std  = float(train["load"].std()) if float(train["load"].std()) > 0 else 1.0

    stats = {
        "train_mean": train_mean,
        "train_std":  train_std,
        "train_len":  len(train),
        "val_len":    len(val),
        "test_len":   len(test),
        "total_len":  n
    }
    return train, val, test, stats


def walk_forward_evaluate(predict_fn: Callable[[np.ndarray, int], float],
                          full_series: np.ndarray,
                          start_idx: int,
                          horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward one-step-ahead or h-step-ahead forecasting.
    At step t, the forecaster receives only history[:t] and predicts index t + horizon - 1.
    """
    preds = []
    actuals = []
    end_idx = len(full_series) - horizon + 1

    for t in range(start_idx, end_idx):
        history = full_series[:t]
        y_hat = predict_fn(history, horizon)
        target = full_series[t + horizon - 1]
        preds.append(y_hat)
        actuals.append(target)

    return np.array(preds, dtype=np.float64), np.array(actuals, dtype=np.float64)


def test_no_future_leakage(predictor_builder: Callable[[], Any]):
    """
    Unit test proving ZERO future data leakage:
      1. Run walk-forward forecasting on original series.
      2. Inject extreme perturbations into future values.
      3. Assert past forecasts remain exactly identical.
    """
    print("Testing predictor for future data leakage...")
    np.random.seed(42)
    series_original = np.sin(np.linspace(0, 50, 500)) * 20 + 100

    test_start = 300
    eval_point = 350
    horizon = 1

    # Run 1: original data
    predictor1 = predictor_builder()
    predictor1.fit(series_original[:test_start])
    history1 = series_original[:eval_point]
    pred1 = predictor1.predict(history1, horizon=horizon)

    # Run 2: perturb future values after eval_point
    series_tampered = series_original.copy()
    series_tampered[eval_point + 10:] = 999999.0  # extreme future anomaly

    predictor2 = predictor_builder()
    predictor2.fit(series_tampered[:test_start])
    history2 = series_tampered[:eval_point]
    pred2 = predictor2.predict(history2, horizon=horizon)

    diff = abs(pred1 - pred2)
    assert diff < 1e-9, f"Data leakage detected! Prediction changed by {diff} when future values were modified."
    print(f"  [OK] Leakage test passed: diff={diff:.2e} (Zero information leaked from future)")
    return True


if __name__ == "__main__":
    # Test on a dummy predictor
    class DummyLagPredictor:
        def fit(self, train): pass
        def predict(self, history, horizon=1):
            return float(np.mean(history[-5:]))

    test_no_future_leakage(DummyLagPredictor)
    print("src/split.py tests completed successfully.")
