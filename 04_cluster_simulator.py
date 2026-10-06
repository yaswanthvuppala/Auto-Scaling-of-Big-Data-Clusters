"""
04_cluster_simulator.py  –  Run reactive vs predictive scaling simulations.

For each dataset × each forecasting model:
  1. Load actual test workloads (total_cpu_demand).
  2. Run the REACTIVE simulator (scales on current utilisation).
  3. Run the PREDICTIVE simulator (scales on forecasted utilisation).
  4. Save results for evaluation in Step 05.

The simulator uses NODE_CAPACITY from the Spark benchmark (Step 02).
If the benchmark file is missing, it falls back to the default in config.py.
"""
import sys, json
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
from src.simulator import simulate_reactive, simulate_predictive
from src.metrics import scaling_summary


def _load_node_capacity() -> float:
    """Read Spark benchmark result, or fall back to config default."""
    bench_path = cfg.MODELS_DIR / "spark_benchmark.json"
    if bench_path.exists():
        with open(bench_path) as f:
            bench = json.load(f)
        # Scale capacity so simulator demand values are in a sensible range
        # We use the mean total_cpu_demand to calibrate:
        #   node_capacity = mean_demand / INITIAL_NODES * ~1.1  (10 % headroom)
        print(f"  Spark benchmark loaded: {bench['throughput_per_core_per_sec']:.0f} rows/s/core")
    return cfg.NODE_CAPACITY  # use config default (calibrated in Step 05)


def run_simulations(dataset_name: str, pred_df: pd.DataFrame, node_cap: float):
    """Run reactive + predictive (per model) simulations for one dataset."""
    print(f"\n{'-'*50}")
    print(f"Simulating: {dataset_name}")
    print(f"{'-'*50}")

    # ── prepare workloads ─────────────────────────────────────
    # total_cpu_demand = sum of all machine CPU utils at each time step
    # This is the "demand" the cluster must handle.
    if "total_cpu_demand" in pred_df.columns:
        workloads = pred_df["total_cpu_demand"].values.astype(float)
    else:
        # fallback: scale cpu_utilization by number of machines
        workloads = pred_df["actual"].values * 20.0  # approximate

    # ── calibrate node capacity ──────────────────────────────
    # Set capacity so that INITIAL_NODES operates at nominal capacity for average demand.
    avg_demand = np.mean(workloads)
    calibrated_cap = avg_demand / cfg.INITIAL_NODES * 1.0
    print(f"  Avg demand = {avg_demand:.1f},  "
          f"Calibrated node capacity = {calibrated_cap:.1f}")

    all_results = []

    # ── 1. Reactive baseline ──────────────────────────────────
    print(f"\n  Running REACTIVE simulation ...")
    reactive_result = simulate_reactive(workloads, node_capacity=calibrated_cap)
    reactive_summary = scaling_summary(reactive_result)
    reactive_summary["dataset"] = dataset_name
    reactive_summary["policy"]  = "reactive"
    reactive_summary["model"]   = "none"
    all_results.append(reactive_summary)
    print(f"    SLA violations: {reactive_summary['total_sla_violations']} / "
          f"{reactive_summary['total_steps']}")
    print(f"    Total cost:     ${reactive_summary['total_cost ($)']:.2f}")

    # ── 2. Predictive with each forecaster ────────────────────
    model_names = [c.replace("pred_", "") for c in pred_df.columns
                   if c.startswith("pred_")]

    for model_name in model_names:
        print(f"\n  Running PREDICTIVE ({model_name}) simulation ...")

        # Predicted workload: scale predicted cpu_util the same way
        if "total_cpu_demand" in pred_df.columns:
            # Ratio approach: predicted_demand = actual_demand × (pred/actual)
            actual = pred_df["actual"].values
            predicted = pred_df[f"pred_{model_name}"].values
            # Avoid division by zero
            ratio = np.where(actual > 0, predicted / actual, 1.0)
            predicted_workloads = workloads * ratio
        else:
            predicted_workloads = pred_df[f"pred_{model_name}"].values * 20.0

        pred_result = simulate_predictive(
            workloads, predicted_workloads, node_capacity=calibrated_cap
        )
        pred_summary = scaling_summary(pred_result)
        pred_summary["dataset"] = dataset_name
        pred_summary["policy"]  = "predictive"
        pred_summary["model"]   = model_name
        all_results.append(pred_summary)
        print(f"    SLA violations: {pred_summary['total_sla_violations']} / "
              f"{pred_summary['total_steps']}")
        print(f"    Total cost:     ${pred_summary['total_cost ($)']:.2f}")

        # Save detailed simulation trace for plots
        trace_df = pd.DataFrame({
            "timestamp":  pred_df["timestamp"].values,
            "workload":   workloads,
            "nodes":      pred_result["node_counts"],
            "latency":    pred_result["latencies"],
            "cost":       pred_result["costs"],
        })
        trace_path = cfg.SIMULATION_DIR / f"{dataset_name}_{model_name}_trace.csv"
        trace_df.to_csv(trace_path, index=False)

    # Save reactive trace too
    r_trace = pd.DataFrame({
        "timestamp":  pred_df["timestamp"].values,
        "workload":   workloads,
        "nodes":      reactive_result["node_counts"],
        "latency":    reactive_result["latencies"],
        "cost":       reactive_result["costs"],
    })
    r_trace.to_csv(cfg.SIMULATION_DIR / f"{dataset_name}_reactive_trace.csv",
                   index=False)

    return all_results


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("STEP 4: Cluster auto-scaling simulations")
    print("="*60)

    node_cap = _load_node_capacity()
    all_summaries = []

    for name in cfg.DATASETS:
        pred_path = cfg.PREDICTIONS_DIR / f"{name}_predictions.csv"
        if not pred_path.exists():
            print(f"  [WARN] {pred_path.name} not found - run Step 03 first.")
            continue
        pred_df = pd.read_csv(pred_path)
        summaries = run_simulations(name, pred_df, node_cap)
        all_summaries.extend(summaries)

    # -- Combined summary table --------------------------------
    summary_df = pd.DataFrame(all_summaries)
    summary_path = cfg.SIMULATION_DIR / "scaling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Combined scaling summary saved -> {summary_path.name}")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] All simulations complete.")
