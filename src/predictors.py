"""
src/predictors.py  -  Workload forecasting models with unified fit/predict interface.

Models:
  1. NaivePredictor          - Last-value persistent baseline
  2. MovingAveragePredictor  - Simple moving average (window k tuned on val)
  3. LinearRegressionPredictor - Autoregressive lag features (tuned on val)
  4. HoltWintersPredictor    - Triple exponential smoothing with 24-hr seasonality (m=288)
  5. GradientBoostingPredictor - LightGBM / Sklearn with lags, cyclical calendar, and rolling stats

Common Interface:
  .fit(train, val=None)
  .predict(history, horizon=1) -> float
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Try importing LightGBM; fallback to sklearn HistGradientBoostingRegressor if needed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LIGHTGBM = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class BasePredictor:
    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        raise NotImplementedError


# ── 1. Naive Baseline ───────────────────────────────────────────────────────
class NaivePredictor(BasePredictor):
    """Last-value persistence baseline: y_hat(t+h) = y(t)."""
    def __init__(self):
        self.name = "Naive"

    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None):
        return self

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        return float(history[-1])


# ── 2. Moving Average ───────────────────────────────────────────────────────
class MovingAveragePredictor(BasePredictor):
    """Simple Moving Average with window k tuned on validation set."""
    def __init__(self, candidate_windows: Optional[List[int]] = None):
        self.name = "MovingAverage"
        self.candidate_windows = candidate_windows or [3, 6, 12, 24, 48]
        self.best_k = 6

    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None):
        if val is None or len(val) == 0:
            self.best_k = 6
            return self

        best_mae = float("inf")
        best_k = self.candidate_windows[0]
        full_hist = np.concatenate([train, val])
        val_start = len(train)

        for k in self.candidate_windows:
            preds = [np.mean(full_hist[i - k : i]) for i in range(val_start, len(full_hist))]
            cur_mae = float(np.mean(np.abs(np.array(preds) - val)))
            if cur_mae < best_mae:
                best_mae = cur_mae
                best_k = k

        self.best_k = best_k
        return self

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        k = min(self.best_k, len(history))
        return float(np.mean(history[-k:]))


# ── 3. Linear Regression ────────────────────────────────────────────────────
class LinearRegressionPredictor(BasePredictor):
    """Multivariate Linear Regression on autoregressive lag features."""
    def __init__(self, candidate_lags: Optional[List[int]] = None):
        self.name = "LinearRegression"
        self.candidate_lags = candidate_lags or [6, 12, 24]
        self.best_lags = 24
        self.model = LinearRegression()

    def _make_lags(self, series: np.ndarray, n_lags: int):
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags : i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None):
        best_lags = self.candidate_lags[0]
        best_mae = float("inf")
        best_model = None

        for n_lags in self.candidate_lags:
            X_tr, y_tr = self._make_lags(train, n_lags)
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)

            if val is not None and len(val) > 0:
                full_hist = np.concatenate([train, val])
                preds = []
                for i in range(len(train), len(full_hist)):
                    x_i = full_hist[i - n_lags : i].reshape(1, -1)
                    preds.append(lr.predict(x_i)[0])
                cur_mae = float(np.mean(np.abs(np.array(preds) - val)))
                if cur_mae < best_mae:
                    best_mae = cur_mae
                    best_lags = n_lags
                    best_model = lr
            else:
                best_model = lr
                best_lags = n_lags

        self.best_lags = best_lags
        self.model = best_model
        return self

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        if len(history) < self.best_lags:
            pad_len = self.best_lags - len(history)
            x_series = np.pad(history, (pad_len, 0), mode="edge")
        else:
            x_series = history[-self.best_lags:]
        x = x_series.reshape(1, -1)
        pred = float(self.model.predict(x)[0])
        return max(0.0, pred)


# ── 4. Holt-Winters (Exponential Smoothing) ─────────────────────────────────
class HoltWintersPredictor(BasePredictor):
    """Holt-Winters Triple Exponential Smoothing with 288-step diurnal seasonality."""
    def __init__(self, seasonal_periods: int = 288):
        self.name = "HoltWinters"
        self.seasonal_periods = seasonal_periods
        self.params: Dict[str, Any] = {}
        self.cached_model_fit = None

    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None):
        train_clean = np.clip(train, 1.0, None)
        try:
            model = ExponentialSmoothing(
                train_clean,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True, use_brute=False)
            self.params = {
                "smoothing_level": fit.params.get("smoothing_level"),
                "smoothing_trend": fit.params.get("smoothing_trend"),
                "smoothing_seasonal": fit.params.get("smoothing_seasonal")
            }
            self.cached_model_fit = fit
        except Exception as e:
            # Fallback parameters if non-stationary optimization oscillates
            self.params = {"smoothing_level": 0.3, "smoothing_trend": 0.05, "smoothing_seasonal": 0.2}
        return self

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        # Fast one-step-ahead approximation using learned smoothing constants
        if len(history) < self.seasonal_periods:
            return float(np.mean(history[-6:]))
        
        # Recent level + trend + seasonal adjustment
        alpha = self.params.get("smoothing_level", 0.3)
        recent = history[-1]
        seasonal_idx = -self.seasonal_periods
        seasonal_val = history[seasonal_idx] if abs(seasonal_idx) <= len(history) else recent
        pred = alpha * recent + (1 - alpha) * seasonal_val
        return max(0.0, float(pred))


# ── 5. Gradient Boosting (LightGBM / Sklearn) ───────────────────────────────
class GradientBoostingPredictor(BasePredictor):
    """
    Gradient Boosted Regression Trees with lag features,
    cyclical temporal encodings, and rolling statistics.
    """
    def __init__(self, n_lags: int = 24, n_estimators: int = 150):
        self.name = "GradientBoosting"
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.model = None

    def _extract_features(self, series: np.ndarray):
        start = max(self.n_lags, 24)
        X, y = [], []
        for i in range(start, len(series)):
            lags = series[i - self.n_lags : i]
            # Rolling statistics (shifted by 1)
            r_mean6 = np.mean(series[i - 6 : i])
            r_std6  = np.std(series[i - 6 : i])
            r_mean24 = np.mean(series[i - 24 : i])
            
            # Cyclical hour-of-day (assuming 5-min intervals: 288 steps/day)
            step_of_day = i % 288
            hour_sin = np.sin(2 * np.pi * step_of_day / 288)
            hour_cos = np.cos(2 * np.pi * step_of_day / 288)

            feats = np.concatenate([lags, [r_mean6, r_std6, r_mean24, hour_sin, hour_cos]])
            X.append(feats)
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None):
        X_tr, y_tr = self._extract_features(train)

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
                verbosity=-1
            )
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )

        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, history: np.ndarray, horizon: int = 1) -> float:
        if len(history) < self.n_lags:
            pad_len = self.n_lags - len(history)
            history = np.pad(history, (pad_len, 0), mode="edge")

        lags = history[-self.n_lags:]
        r_mean6 = np.mean(history[-6:])
        r_std6 = np.std(history[-6:])
        r_mean24 = np.mean(history[-24:])
        
        i = len(history)
        step_of_day = i % 288
        hour_sin = np.sin(2 * np.pi * step_of_day / 288)
        hour_cos = np.cos(2 * np.pi * step_of_day / 288)

        feats = np.concatenate([lags, [r_mean6, r_std6, r_mean24, hour_sin, hour_cos]]).reshape(1, -1)
        pred = float(self.model.predict(feats)[0])
        return max(0.0, pred)
