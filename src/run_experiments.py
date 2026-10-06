"""
src/run_experiments.py  -  Full experimental grid evaluation and statistical aggregation.

Grid Dimensions:
  - Datasets: D1 (Alibaba 2018), D2 (Azure 2019), D3 (Alibaba GPU 2020)
  - Policies: Reactive, Naive, MA, LinearReg, HoltWinters, GBM, Hybrid
  - Provisioning Delays: 1, 3, 5, 10 time steps (5, 15, 25, 50 minutes)
  - Safety Margins: 0%, 10%, 20% (0.0, 0.10, 0.20)
  - Statistical Robustness: 5 non-overlapping test chunks + boot-time jitter (mean +/- std)
  - Output: results/tables/all_results.csv, results/tables/forecasting_results.csv, and publication plots
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.split import train_val_test_split, walk_forward_evaluate
from src.simulator import ClusterSimulator
from src.predictors import (
    NaivePredictor,
    MovingAveragePredictor,
    LinearRegressionPredictor,
    HoltWintersPredictor,
    GradientBoostingPredictor
)
from src.scalers import ReactiveScaler, PredictiveScaler, HybridScaler
from src.metrics import compute_forecast_metrics, compute_scaling_metrics, aggregate_runs_mean_std
from src.plots import generate_experiment_plots, generate_eda_plots

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def run_all_experiments():
    print("=" * 80)
    print("STEP 14: FULL EXPERIMENT GRID (POLICIES x DATASETS x DELAYS x MARGINS)")
    print("=" * 80)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = ["D1", "D2", "D3"]
    delays = [1, 3, 5, 10]
    safety_margins = [0.0, 0.10, 0.20]
    n_test_chunks = 5  # 5 non-overlapping sub-windows for mean +/- std

    all_sim_records = []
    forecasting_records = []

    for d_name in dataset_names:
        csv_path = PROCESSED_DIR / f"{d_name}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found. Run `python src/preprocess_spark.py` first.")
            continue

        print(f"\n{'#' * 60}")
        print(f"# Processing Dataset: {d_name}")
        print(f"{'#' * 60}")

        df = pd.read_csv(csv_path)
        train_df, val_df, test_df, stats = train_val_test_split(df, 0.60, 0.20, 0.20)

        train_y = train_df["load"].values
        val_y = val_df["load"].values
        test_y = test_df["load"].values
        full_series = np.concatenate([train_y, val_y, test_y])
        test_start_idx = len(train_y) + len(val_y)

        # 1. Fit and tune predictors on train/val
        print("\n1. Fitting and tuning predictors on train/validation sets...")
        predictors = {
            "Naive": NaivePredictor().fit(train_y, val_y),
            "MovingAverage": MovingAveragePredictor().fit(train_y, val_y),
            "LinearRegression": LinearRegressionPredictor().fit(train_y, val_y),
            "HoltWinters": HoltWintersPredictor().fit(train_y, val_y),
            "GradientBoosting": GradientBoostingPredictor().fit(train_y, val_y)
        }

        # 2. Evaluate forecasting metrics on test set
        print("2. Computing walk-forward test forecasting metrics...")
        for p_name, predictor in predictors.items():
            preds, targets = walk_forward_evaluate(
                lambda hist, h: predictor.predict(hist, horizon=h),
                full_series,
                start_idx=test_start_idx,
                horizon=1
            )
            fc_metrics = compute_forecast_metrics(targets, preds)
            fc_metrics["dataset"] = d_name
            fc_metrics["model"] = p_name
            forecasting_records.append(fc_metrics)
            print(f"  {p_name:18s} -> MAE: {fc_metrics['MAE']:.2f} | RMSE: {fc_metrics['RMSE']:.2f} | MAPE: {fc_metrics['MAPE (%)']:.2f}%")

        # 3. Simulate scaling policies across grid
        print("\n3. Executing cluster simulations across experimental grid...")
        chunk_len = len(test_y) // n_test_chunks

        # Calculate total simulation runs
        total_iterations = len(delays) * len(safety_margins) * n_test_chunks * 7  # 7 policies
        pbar = tqdm(total=total_iterations, desc=f"Simulating {d_name}", unit="run")

        for delay in delays:
            for margin in safety_margins:
                for chunk_idx in range(n_test_chunks):
                    # Sub-window slice
                    s_idx = chunk_idx * chunk_len
                    e_idx = (chunk_idx + 1) * chunk_len
                    chunk_workload = test_y[s_idx:e_idx]

                    # Simulator instance
                    sim = ClusterSimulator(
                        node_capacity=100.0,
                        cost_per_node_minute=0.002,
                        interval_minutes=5,
                        provisioning_delay_steps=delay,
                        min_nodes=2,
                        max_nodes=50,
                        scale_in_cooldown_steps=3,
                        sla_latency_limit=1.0,
                        initial_nodes=10
                    )

                    # Build policies with historical context
                    start_offset = test_start_idx + s_idx
                    policies = {
                        "Reactive": ReactiveScaler(scale_up_thresh=0.80, scale_down_thresh=0.40),
                        "Predictive-Naive": PredictiveScaler(predictors["Naive"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                        "Predictive-MovingAverage": PredictiveScaler(predictors["MovingAverage"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                        "Predictive-LinearRegression": PredictiveScaler(predictors["LinearRegression"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                        "Predictive-HoltWinters": PredictiveScaler(predictors["HoltWinters"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                        "Predictive-GradientBoosting": PredictiveScaler(predictors["GradientBoosting"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                        "Hybrid-GradientBoosting": HybridScaler(predictors["GradientBoosting"], full_series, start_offset=start_offset, horizon=delay, safety_margin=margin),
                    }

                    for p_label, policy in policies.items():
                        # Run simulation with small boot-time jitter (std = 0.2 steps)
                        sim_res = sim.run(chunk_workload, policy, boot_jitter_std=0.2)
                        m = compute_scaling_metrics(sim_res, interval_minutes=5, cost_per_node_minute=0.002)

                        m["dataset"] = d_name
                        m["policy"] = p_label
                        m["provisioning_delay"] = delay
                        m["safety_margin"] = margin
                        m["chunk_idx"] = chunk_idx
                        all_sim_records.append(m)
                        pbar.update(1)

        pbar.close()

    # Save forecasting results
    fc_df = pd.DataFrame(forecasting_records)
    out_fc = TABLES_DIR / "forecasting_results.csv"
    fc_df.to_csv(out_fc, index=False)
    print(f"\n[OK] Forecasting results saved to {out_fc}")
    print(fc_df.to_string(index=False))

    # Save detailed simulation results
    sim_df = pd.DataFrame(all_sim_records)
    out_all = TABLES_DIR / "all_results.csv"
    sim_df.to_csv(out_all, index=False)
    print(f"\n[OK] All experimental runs saved to {out_all}")

    # Generate aggregated statistical summary table (mean +/- std)
    print("\n" + "=" * 80)
    print("AGGREGATED PERFORMANCE SUMMARY (MEAN +/- STD OVER 5 TEST RUNS)")
    print("=" * 80)
    
    # Filter for standard representative baseline (delay=1, margin=0.10)
    rep_df = sim_df[(sim_df["provisioning_delay"] == 1) & (sim_df["safety_margin"] == 0.10)]
    agg_rows = []
    for (d, p), group in rep_df.groupby(["dataset", "policy"]):
        agg_rows.append({
            "Dataset": d,
            "Policy": p,
            "SLA Viol. (%)": f"{group['sla_violation_rate_pct'].mean():.2f} +/- {group['sla_violation_rate_pct'].std():.2f}",
            "Cost ($)": f"{group['total_cost_usd'].mean():.2f} +/- {group['total_cost_usd'].std():.2f}",
            "P95 Latency": f"{group['p95_latency'].mean():.2f} +/- {group['p95_latency'].std():.2f}",
            "Avg Util (%)": f"{group['avg_utilization_pct'].mean():.1f}% +/- {group['avg_utilization_pct'].std():.1f}%",
            "Scaling Actions": f"{group['scaling_actions'].mean():.1f} +/- {group['scaling_actions'].std():.1f}"
        })
    agg_table = pd.DataFrame(agg_rows)
    out_agg = TABLES_DIR / "aggregated_summary.csv"
    agg_table.to_csv(out_agg, index=False)
    print(agg_table.to_string(index=False))
    print(f"\n[OK] Aggregated summary saved to {out_agg}")

    # Step 15: Render all report figures
    generate_experiment_plots(out_all)


if __name__ == "__main__":
    run_all_experiments()
