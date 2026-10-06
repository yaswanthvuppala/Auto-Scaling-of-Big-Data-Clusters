"""
forecasters.py  –  Four workload-forecasting models, simplest → most complex.

Each class follows the same interface:
    .fit(train_series, val_series)   → learns parameters
    .predict(history)                → returns next predicted value
    .walk_forward(full_series, start)→ one-step-ahead predictions from 'start'

Why these four?
  1. Moving Average (MA)     — baseline; no learning, just smoothing.
  2. Linear Regression (LR)  — learns a linear relationship between lags & future.
  3. Holt-Winters (HW)       — classical time-series model with trend + seasonality.
  4. Gradient Boosting (GB)   — modern ML; captures non-linear patterns.

Walk-forward forecasting (used in evaluate):
  At every test step t, the model sees *actual* values up to t−1 and predicts t.
  Then we move one step forward. This mirrors real-world deployment: the
  auto-scaler always knows the current load but must predict the next interval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import config as cfg


# ═══════════════════════════════════════════════════════════════
# 1.  MOVING AVERAGE
# ═══════════════════════════════════════════════════════════════
class MovingAverageForecaster:
    """
    Predicts the next value as the mean of the last W observations.
    The best window W is chosen on the validation set.

    Viva Q: "Why is this a good baseline?"
    A: It adapts to recent trends without any training; it is the simplest
       smoother and gives a lower bound on performance.
    """
    def __init__(self, candidate_windows=None):
        self.candidate_windows = candidate_windows or cfg.MA_WINDOWS
        self.best_window = None

    def fit(self, train: np.ndarray, val: np.ndarray):
        """Try each candidate window; pick the one with lowest MAE on val."""
        best_mae, best_w = np.inf, self.candidate_windows[0]
        full = np.concatenate([train, val])
        val_start = len(train)

        for w in self.candidate_windows:
            preds = []
            for i in range(val_start, len(full)):
                preds.append(np.mean(full[i - w : i]))
            cur_mae = np.mean(np.abs(np.array(preds) - val))
            if cur_mae < best_mae:
                best_mae, best_w = cur_mae, w

        self.best_window = best_w
        print(f"    MA best window = {best_w}  (val MAE = {best_mae:.4f})")

    def predict_next(self, history: np.ndarray) -> float:
        """Given history up to t−1, predict t."""
        return float(np.mean(history[-self.best_window :]))

    def walk_forward(self, full_series: np.ndarray, start_idx: int) -> np.ndarray:
        """One-step-ahead predictions from start_idx to end."""
        preds = []
        for i in range(start_idx, len(full_series)):
            preds.append(self.predict_next(full_series[:i]))
        return np.array(preds)

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


# ═══════════════════════════════════════════════════════════════
# 2.  LINEAR REGRESSION  (with lag + calendar features)
# ═══════════════════════════════════════════════════════════════
class LinearRegressionForecaster:
    """
    Treats forecasting as supervised regression:
        features = [lag_1, lag_2, …, lag_L, hour_sin, hour_cos, dow_sin, dow_cos]
        target   = cpu_utilization at time t

    Viva Q: "Why lag features instead of raw time?"
    A: The auto-correlation in workloads means recent values are the strongest
       predictors of the next value (temporal locality).
    """
    def __init__(self, n_lags=None):
        self.n_lags = n_lags or cfg.LR_LAGS
        self.model = LinearRegression()

    def _build_features(self, series, timestamps=None):
        """Build lag matrix; optionally add hour/dow from timestamps."""
        from src.utils import create_lag_features
        X, y = create_lag_features(series, self.n_lags)
        if timestamps is not None:
            ts = pd.to_datetime(timestamps[self.n_lags:])
            time_feats = np.column_stack([
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
                np.sin(2 * np.pi * ts.dayofweek / 7),
                np.cos(2 * np.pi * ts.dayofweek / 7),
            ])
            X = np.hstack([X, time_feats])
        return X, y

    def fit(self, train: np.ndarray, val: np.ndarray, timestamps=None):
        X_tr, y_tr = self._build_features(train, timestamps[:len(train)] if timestamps is not None else None)
        self.model.fit(X_tr, y_tr)
        # report val MAE
        full = np.concatenate([train, val])
        ts_full = timestamps if timestamps is not None else None
        preds = self.walk_forward(full, len(train), ts_full)
        val_mae = np.mean(np.abs(preds - val))
        print(f"    LR  n_lags = {self.n_lags}  (val MAE = {val_mae:.4f})")

    def predict_next(self, history: np.ndarray, timestamp=None) -> float:
        x = history[-self.n_lags:].reshape(1, -1)
        if timestamp is not None:
            ts = pd.Timestamp(timestamp)
            tf = np.array([[
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
                np.sin(2 * np.pi * ts.dayofweek / 7),
                np.cos(2 * np.pi * ts.dayofweek / 7),
            ]])
            x = np.hstack([x, tf])
        return float(self.model.predict(x)[0])

    def walk_forward(self, full_series, start_idx, timestamps=None):
        preds = []
        for i in range(start_idx, len(full_series)):
            ts = timestamps[i] if timestamps is not None else None
            preds.append(self.predict_next(full_series[:i], ts))
        return np.array(preds)

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


# ═══════════════════════════════════════════════════════════════
# 3.  HOLT-WINTERS  (Triple Exponential Smoothing)
# ═══════════════════════════════════════════════════════════════
class HoltWintersForecaster:
    """
    Classical time-series decomposition into three components:
        Level  (l_t):       smoothed average
        Trend  (b_t):       direction of change
        Season (s_t):       repeating pattern (period = 1 day = 288 steps)

    Forecast:  ŷ(t+1) = (l_t + b_t) × s_{t+1−m}

    Viva Q: "Why seasonal_periods = 288?"
    A: At 5-min resolution, one day = 24×12 = 288 intervals.
       Workloads repeat daily (peak during business hours, low at night).
    """
    def __init__(self, seasonal_periods=None):
        self.seasonal_periods = seasonal_periods or cfg.HW_SEASONAL_PERIODS
        self.model_fit = None

    def fit(self, train: np.ndarray, val: np.ndarray):
        # Clamp small values to avoid log(0) in multiplicative mode
        train_clean = np.clip(train, 1.0, None)
        try:
            model = ExponentialSmoothing(
                train_clean,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.model_fit = model.fit(optimized=True, use_brute=False)
            # Evaluate on val
            preds = self.walk_forward_from_fit(train_clean, val)
            val_mae = np.mean(np.abs(preds - val))
            print(f"    HW  seasonal_periods = {self.seasonal_periods}  "
                  f"(val MAE = {val_mae:.4f})")
        except Exception as e:
            print(f"    HW fitting failed: {e}; using fallback (MA-48)")
            self.model_fit = None

    def walk_forward_from_fit(self, train, val):
        """Re-fit on train, then predict val length ahead."""
        preds = self.model_fit.forecast(len(val))
        return np.array(preds)

    def walk_forward(self, full_series, start_idx, timestamps=None):
        """Walk-forward one-step-ahead forecasting: fit on train/val,
        then recursively filter test set using learned parameters."""
        train_data = np.clip(full_series[:start_idx], 1.0, None)
        try:
            model = ExponentialSmoothing(
                train_data,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=False)

            full_clean = np.clip(full_series, 1.0, None)
            full_model = ExponentialSmoothing(
                full_clean,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
            )
            full_fit = full_model.fit(
                smoothing_level=fit.params.get("smoothing_level"),
                smoothing_trend=fit.params.get("smoothing_trend"),
                smoothing_seasonal=fit.params.get("smoothing_seasonal"),
                optimized=False,
            )
            return np.array(full_fit.fittedvalues[start_idx:])
        except Exception as e:
            # Fallback: use MA
            preds = []
            for i in range(start_idx, len(full_series)):
                preds.append(np.mean(full_series[max(0, i-48):i]))
            return np.array(preds)

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


# ═══════════════════════════════════════════════════════════════
# 4.  GRADIENT BOOSTING  (with lag + calendar + rolling features)
# ═══════════════════════════════════════════════════════════════
class GradientBoostingForecaster:
    """
    Ensemble of shallow decision trees, each correcting the error of the
    previous ones.  Uses HistGradientBoostingRegressor (fast, handles NaNs).

    Feature set (richest of all four models):
        - lag_1 … lag_L
        - hour_sin, hour_cos, dow_sin, dow_cos
        - rolling_mean_W, rolling_std_W  for each window W

    Viva Q: "Why gradient boosting over a neural network?"
    A: For tabular data with <100 features, gradient boosting typically
       matches or beats neural nets, trains in seconds, and is easier to
       interpret (feature importances).
    """
    def __init__(self, n_lags=None, rolling_windows=None):
        self.n_lags = n_lags or cfg.GB_LAGS
        self.rolling_windows = rolling_windows or cfg.GB_ROLLING_WINDOWS
        self.model = HistGradientBoostingRegressor(
            max_iter=cfg.GB_N_ESTIMATORS,
            max_depth=cfg.GB_MAX_DEPTH,
            learning_rate=cfg.GB_LEARNING_RATE,
            random_state=cfg.RANDOM_SEED,
        )
        self.feature_names = None

    def _build_features(self, series, timestamps=None):
        """Build full feature matrix from raw series."""
        n_lags = self.n_lags
        max_roll = max(self.rolling_windows) if self.rolling_windows else 0
        start = max(n_lags, max_roll + 1)  # first valid index

        rows = []
        for i in range(start, len(series)):
            row = {}
            # lags
            for lag in range(1, n_lags + 1):
                row[f"lag_{lag}"] = series[i - lag]
            # calendar
            if timestamps is not None:
                ts = pd.Timestamp(timestamps[i])
                row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
                row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
                row["dow_sin"]  = np.sin(2 * np.pi * ts.dayofweek / 7)
                row["dow_cos"]  = np.cos(2 * np.pi * ts.dayofweek / 7)
            # rolling stats
            for w in self.rolling_windows:
                window = series[i - w - 1 : i - 1]  # shifted by 1 to avoid leakage
                if len(window) >= w:
                    row[f"roll_mean_{w}"] = np.mean(window)
                    row[f"roll_std_{w}"]  = np.std(window)
                else:
                    row[f"roll_mean_{w}"] = np.nan
                    row[f"roll_std_{w}"]  = np.nan
            rows.append(row)

        X = pd.DataFrame(rows)
        y = series[start:]
        self.feature_names = list(X.columns)
        return X.values, y

    def fit(self, train, val, timestamps=None):
        ts_train = timestamps[:len(train)] if timestamps is not None else None
        X_tr, y_tr = self._build_features(train, ts_train)
        self.model.fit(X_tr, y_tr)

        # val MAE
        full = np.concatenate([train, val])
        ts_full = timestamps if timestamps is not None else None
        preds = self.walk_forward(full, len(train), ts_full)
        val_mae = np.mean(np.abs(preds - val))
        print(f"    GB  n_lags={self.n_lags}, trees={cfg.GB_N_ESTIMATORS}  "
              f"(val MAE = {val_mae:.4f})")

    def predict_next(self, history, timestamp=None):
        n_lags = self.n_lags
        row = {}
        for lag in range(1, n_lags + 1):
            row[f"lag_{lag}"] = history[-lag] if lag <= len(history) else 0.0
        if timestamp is not None:
            ts = pd.Timestamp(timestamp)
            row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
            row["dow_sin"]  = np.sin(2 * np.pi * ts.dayofweek / 7)
            row["dow_cos"]  = np.cos(2 * np.pi * ts.dayofweek / 7)
        for w in self.rolling_windows:
            if len(history) >= w + 1:
                window = history[-(w+1):-1]
                row[f"roll_mean_{w}"] = np.mean(window)
                row[f"roll_std_{w}"]  = np.std(window)
            else:
                row[f"roll_mean_{w}"] = np.mean(history)
                row[f"roll_std_{w}"]  = np.std(history)
        x = np.array([[row.get(f, 0.0) for f in self.feature_names]])
        return float(self.model.predict(x)[0])

    def walk_forward(self, full_series, start_idx, timestamps=None):
        preds = []
        for i in range(start_idx, len(full_series)):
            ts = timestamps[i] if timestamps is not None else None
            preds.append(self.predict_next(full_series[:i], ts))
        return np.array(preds)

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


# ─────────────── convenience registry ──────────────────────────
FORECASTER_CLASSES = {
    "MA":  MovingAverageForecaster,
    "LR":  LinearRegressionForecaster,
    "HW":  HoltWintersForecaster,
    "GB":  GradientBoostingForecaster,
}
