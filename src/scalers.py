"""
src/scalers.py  -  Auto-scaling policies: Reactive, Predictive, and Hybrid.

Policies:
  1. ReactiveScaler:
     - Thresholds: scale-out at util > 80%, scale-in at util < 40%.
     - Evaluates current workload demand at step t.
  2. PredictiveScaler:
     - Forecasts workload at t+h (h >= provisioning delay).
     - Proactively provisions nodes with configurable safety_margin (0%, 10%, 20%).
  3. HybridScaler:
     - Primary proactive scale-out based on model forecast.
     - Secondary reactive safety net triggering if current utilization exceeds emergency threshold.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


class ReactiveScaler:
    """
    Classic threshold-based reactive elasticity controller.
    """
    def __init__(self,
                 scale_up_thresh: float = 0.80,
                 scale_down_thresh: float = 0.40,
                 target_util: float = 0.75):
        self.scale_up_thresh = scale_up_thresh
        self.scale_down_thresh = scale_down_thresh
        self.target_util = target_util
        self.name = "Reactive"

    def __call__(self, step: int, state: Dict[str, Any]) -> int:
        cap = state["capacity"]
        load = state["current_load"] + state["backlog"]
        active = state["active_nodes"]
        node_cap = state["node_capacity"]

        util = load / max(cap, 1e-6)

        if util > self.scale_up_thresh:
            # Scale out proportionally to bring utilization back to target
            target = int(np.ceil(load / (node_cap * self.target_util)))
            return max(active + 1, target)
        elif util < self.scale_down_thresh and state["cooldown_remaining"] == 0:
            # Scale in safely
            target = int(np.ceil(load / (node_cap * self.target_util)))
            return max(state["min_nodes"], target)

        return active + state["pending_nodes"]


class PredictiveScaler:
    """
    Workload-forecast-driven predictive elasticity controller.
    """
    def __init__(self,
                 predictor: Any,
                 history_series: np.ndarray,
                 start_offset: int = 0,
                 horizon: int = 1,
                 safety_margin: float = 0.10,
                 target_util: float = 0.75):
        self.predictor = predictor
        self.full_history = history_series
        self.start_offset = start_offset
        self.horizon = max(1, horizon)
        self.safety_margin = safety_margin
        self.target_util = target_util
        self.name = f"Predictive-{getattr(predictor, 'name', 'Model')}"

    def __call__(self, step: int, state: Dict[str, Any]) -> int:
        active = state["active_nodes"]
        node_cap = state["node_capacity"]

        # Feed known history up to current step (no future data leakage)
        current_idx = self.start_offset + step
        history_window = self.full_history[:current_idx + 1]
        pred_load = self.predictor.predict(history_window, horizon=self.horizon)

        # Proactive target node sizing: nodes = ceil(pred / (node_cap * target_util) * (1 + safety_margin))
        effective_demand = (pred_load + state["backlog"]) * (1.0 + self.safety_margin)
        target = int(np.ceil(effective_demand / (node_cap * self.target_util)))

        if target > active + state["pending_nodes"]:
            return target
        elif target < active and state["cooldown_remaining"] == 0:
            return max(state["min_nodes"], target)

        return active + state["pending_nodes"]


class HybridScaler:
    """
    Hybrid controller: Predictive scale-out with a Reactive safety net.
    """
    def __init__(self,
                 predictor: Any,
                 history_series: np.ndarray,
                 start_offset: int = 0,
                 horizon: int = 1,
                 safety_margin: float = 0.10,
                 reactive_emergency_thresh: float = 0.85,
                 target_util: float = 0.75):
        self.predictive_scaler = PredictiveScaler(predictor, history_series, start_offset=start_offset,
                                                  horizon=horizon, safety_margin=safety_margin, target_util=target_util)
        self.reactive_scaler = ReactiveScaler(scale_up_thresh=reactive_emergency_thresh, target_util=target_util)
        self.name = f"Hybrid-{getattr(predictor, 'name', 'Model')}"

    def __call__(self, step: int, state: Dict[str, Any]) -> int:
        pred_target = self.predictive_scaler(step, state)
        react_target = self.reactive_scaler(step, state)
        return max(pred_target, react_target)
