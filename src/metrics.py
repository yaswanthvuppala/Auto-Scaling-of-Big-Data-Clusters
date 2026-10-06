"""
src/metrics.py  -  Evaluation metrics for forecasting accuracy and cluster auto-scaling quality.

Metrics:
  1. Forecasting: MAE, RMSE, MAPE
  2. Auto-scaling:
     - SLA violation rate (%)
     - Mean & P95 latency
     - Total cost ($ and node-minutes)
     - Average cluster utilization (%)
     - Scaling actions count (scale-outs + scale-ins)
     - Over-provisioning (idle node-minutes and wasted capacity fraction)
  3. Statistical aggregation: mean +/- std over multiple experimental runs.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


# ── Forecasting Metrics ─────────────────────────────────────────────────────
def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def compute_forecast_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": round(mae(actual, pred), 4),
        "RMSE": round(rmse(actual, pred), 4),
        "MAPE (%)": round(mape(actual, pred), 2)
    }


# ── Auto-Scaling Metrics ────────────────────────────────────────────────────
def compute_scaling_metrics(sim_df: pd.DataFrame,
                            interval_minutes: int = 5,
                            cost_per_node_minute: float = 0.002,
                            sla_limit: float = 1.0) -> Dict[str, Any]:
    """
    Computes comprehensive cluster elasticity performance metrics from simulation trace.
    """
    latencies = sim_df["latency"].values
    loads = sim_df["load"].values
    capacities = sim_df["capacity"].values
    active_nodes = sim_df["active_nodes"].values

    total_steps = len(sim_df)
    sla_violations = int(np.sum(latencies > sla_limit))
    sla_rate = (sla_violations / total_steps * 100.0) if total_steps > 0 else 0.0

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    total_node_minutes = float(np.sum(active_nodes) * interval_minutes)
    total_dollar_cost = float(total_node_minutes * cost_per_node_minute)

    # Utilization = load / capacity
    utilizations = np.clip(loads / np.maximum(capacities, 1e-6), 0.0, 1.0)
    avg_utilization = float(np.mean(utilizations) * 100.0)

    # Scaling actions = count of changes in active nodes
    node_diffs = np.diff(active_nodes)
    scale_outs = int(np.sum(node_diffs > 0))
    scale_ins = int(np.sum(node_diffs < 0))
    total_actions = scale_outs + scale_ins

    # Over-provisioning: idle capacity = capacity - load
    idle_capacity = np.maximum(0.0, capacities - loads)
    wasted_fraction = float(np.mean(idle_capacity / np.maximum(capacities, 1e-6)) * 100.0)
    idle_node_minutes = float(np.sum(idle_capacity / (capacities / np.maximum(active_nodes, 1))) * interval_minutes)

    return {
        "total_steps": total_steps,
        "sla_violations": sla_violations,
        "sla_violation_rate_pct": round(sla_rate, 2),
        "mean_latency": round(mean_latency, 3),
        "p95_latency": round(p95_latency, 3),
        "total_node_minutes": round(total_node_minutes, 1),
        "total_cost_usd": round(total_dollar_cost, 2),
        "avg_utilization_pct": round(avg_utilization, 2),
        "scaling_actions": total_actions,
        "scale_outs": scale_outs,
        "scale_ins": scale_ins,
        "avg_overprovision_pct": round(wasted_fraction, 2),
        "idle_node_minutes": round(idle_node_minutes, 1),
    }


def aggregate_runs_mean_std(runs_list: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Format multiple experimental runs as mean +/- std strings.
    """
    df = pd.DataFrame(runs_list)
    summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        m = float(np.mean(vals))
        s = float(np.std(vals))
        summary[col] = f"{m:.2f} +/- {s:.2f}"
    return summary
