"""
utils.py  –  Shared helper functions used across pipeline steps.

Key concepts for viva:
  - time_split(): deterministic, chronological split to avoid data leakage.
  - create_lag_features(): turns a 1-D time series into a supervised-learning
    matrix  [x(t-L), ..., x(t-1)]  →  x(t).
  - create_features_with_time(): adds calendar + rolling statistics on top of
    lags for richer models (LR, GB).
"""
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

# ── make project root importable from any script ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


# ─────────────────── chronological split ───────────────────────
def time_split(df: pd.DataFrame):
    """
    Split a *time-sorted* DataFrame into train / val / test.

    Why chronological?  Because in a real cluster we can never peek
    at future workload.  Random splits would leak future information
    into training and inflate accuracy.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    t1 = int(n * cfg.TRAIN_RATIO)                       # end of train
    t2 = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))     # end of val

    train = df.iloc[:t1].copy()
    val   = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()

    print(f"  Split -> train {len(train)}  |  val {len(val)}  |  test {len(test)}")
    return train, val, test


# ─────────────────── simple lag matrix ─────────────────────────
def create_lag_features(series: np.ndarray, n_lags: int):
    """
    Turn  [x0, x1, x2, x3, x4]  with n_lags=3  into:
        X = [[x0,x1,x2],   y = [x3,
             [x1,x2,x3]]        x4]

    Used by the Moving-Average and Linear-Regression forecasters.
    """
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags : i])
        y.append(series[i])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


# ─────────────────── rich feature matrix ───────────────────────
def create_features_with_time(
    df: pd.DataFrame,
    n_lags: int,
    rolling_windows: list = None,
    target_col: str = "cpu_utilization",
):
    """
    Build a feature DataFrame with:
      1. Lag columns:      lag_1 … lag_L  (past values)
      2. Calendar columns: hour-of-day & day-of-week  (cyclical sin/cos)
      3. Rolling stats:    rolling mean & std over several window sizes

    Cyclical encoding avoids a discontinuity at midnight / week boundary:
        hour_sin = sin(2π · hour / 24)
        hour_cos = cos(2π · hour / 24)

    Returns  (X_df, y_series)  with NaN rows removed.
    """
    feat = pd.DataFrame(index=df.index)

    # 1. lag features
    for lag in range(1, n_lags + 1):
        feat[f"lag_{lag}"] = df[target_col].shift(lag)

    # 2. calendar features (cyclical)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        feat["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        feat["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # 3. rolling statistics (shifted by 1 to prevent leakage)
    if rolling_windows:
        shifted = df[target_col].shift(1)
        for w in rolling_windows:
            feat[f"roll_mean_{w}"] = shifted.rolling(w).mean()
            feat[f"roll_std_{w}"]  = shifted.rolling(w).std()

    y = df[target_col].copy()

    # drop incomplete rows (first n_lags + max_window rows will be NaN)
    valid_idx = feat.dropna().index
    return feat.loc[valid_idx], y.loc[valid_idx]


# ─────────────────── I/O helpers ───────────────────────────────
def load_processed(dataset_name: str) -> pd.DataFrame:
    """Load a Spark-processed Parquet file into pandas."""
    path = cfg.DATA_PROCESSED_DIR / f"{dataset_name}_processed.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {dataset_name}: {len(df)} rows, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df
