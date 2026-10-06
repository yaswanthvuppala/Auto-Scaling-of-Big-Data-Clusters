"""
05_evaluate_and_plot.py  –  Compute metrics and generate publication-ready plots.

Produces:
  A) Forecasting accuracy table     (MAE, RMSE, MAPE per model × dataset)
  B) Scaling comparison table       (SLA violations, cost, over-provisioning)
  C) Plots:
      1. Time-series: actual vs predicted workload (one per dataset)
      2. Bar chart: SLA violation rates by policy
      3. Bar chart: Total cost by policy
      4. Scaling trace: workload + node count + latency over time
      5. Forecast error distribution (box plot)
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
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config as cfg
from src.metrics import mae, rmse, mape


# ═══════════════════════════════════════════════════════════════
# A.  FORECASTING ACCURACY
# ═══════════════════════════════════════════════════════════════
def compute_forecasting_metrics():
    """Compute MAE, RMSE, MAPE for every model × dataset."""
    rows = []
    for name in cfg.DATASETS:
        path = cfg.PREDICTIONS_DIR / f"{name}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        actual = df["actual"].values

        model_cols = [c for c in df.columns if c.startswith("pred_")]
        for col in model_cols:
            model_name = col.replace("pred_", "")
            pred = df[col].values
            rows.append({
                "dataset":  name,
                "model":    model_name,
                "MAE":      round(mae(actual, pred), 4),
                "RMSE":     round(rmse(actual, pred), 4),
                "MAPE (%)": round(mape(actual, pred), 2),
            })

    table = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "forecasting_metrics.csv"
    table.to_csv(out, index=False)
    print("\n=== Forecasting Accuracy ===")
    print(table.to_string(index=False))
    return table


# ============================================================
# B.  SCALING COMPARISON
# ============================================================
def load_scaling_summary():
    """Load the summary CSV produced by Step 04."""
    path = cfg.SIMULATION_DIR / "scaling_summary.csv"
    if not path.exists():
        print("  [WARN] scaling_summary.csv not found - run Step 04 first.")
        return None
    df = pd.read_csv(path)
    print("\n=== Scaling Comparison ===")
    print(df.to_string(index=False))
    return df


# ============================================================
# C.  PLOTS
# ============================================================

def plot_forecasts(dataset_name: str, pred_df: pd.DataFrame):
    """Plot 1: actual vs predicted workload (all models, last 2 days)."""
    fig, ax = plt.subplots(figsize=(14, 5))

    timestamps = pd.to_datetime(pred_df["timestamp"])
    actual = pred_df["actual"].values

    # Show last 2 days for clarity (576 points)
    n_show = min(576, len(pred_df))
    sl = slice(-n_show, None)

    ax.plot(timestamps.values[sl], actual[sl], "k-", lw=1.5,
            label="Actual", alpha=0.8)

    colors = {"MA": "#e74c3c", "LR": "#3498db", "HW": "#2ecc71", "GB": "#9b59b6"}
    for col in pred_df.columns:
        if col.startswith("pred_"):
            model = col.replace("pred_", "")
            ax.plot(timestamps.values[sl], pred_df[col].values[sl],
                    "--", lw=1, label=model, color=colors.get(model, "gray"),
                    alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title(f"Workload Forecast — {dataset_name} (last 2 days of test set)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = cfg.PLOTS_DIR / f"{dataset_name}_forecast.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_scaling_trace(dataset_name: str):
    """Plot 4: workload + nodes + latency for reactive vs best predictive."""
    reactive_path = cfg.SIMULATION_DIR / f"{dataset_name}_reactive_trace.csv"
    if not reactive_path.exists():
        return

    reactive = pd.read_csv(reactive_path)

    # Find best predictive trace (lowest latency file)
    best_pred_file = None
    best_violations = float("inf")
    for model in ["GB", "LR", "HW", "MA"]:
        p = cfg.SIMULATION_DIR / f"{dataset_name}_{model}_trace.csv"
        if p.exists():
            t = pd.read_csv(p)
            viol = (t["latency"] > cfg.SLA_LATENCY_LIMIT).sum()
            if viol < best_violations:
                best_violations = viol
                best_pred_file = p
                best_model = model

    if best_pred_file is None:
        return

    predictive = pd.read_csv(best_pred_file)

    # Show last 2 days
    n_show = min(576, len(reactive))
    sl = slice(-n_show, None)
    ts = pd.to_datetime(reactive["timestamp"].values[sl])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Workload
    axes[0].plot(ts, reactive["workload"].values[sl], "k-", lw=1, label="Workload")
    axes[0].set_ylabel("CPU Demand")
    axes[0].set_title(f"Scaling Trace — {dataset_name} (Reactive vs Predictive-{best_model})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Node counts
    axes[1].step(ts, reactive["nodes"].values[sl], "r-", lw=1.5,
                 label="Reactive", where="post")
    axes[1].step(ts, predictive["nodes"].values[sl], "b-", lw=1.5,
                 label=f"Predictive ({best_model})", where="post")
    axes[1].set_ylabel("Active Nodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Latency
    axes[2].plot(ts, reactive["latency"].values[sl], "r-", lw=1, alpha=0.7,
                 label="Reactive")
    axes[2].plot(ts, predictive["latency"].values[sl], "b-", lw=1, alpha=0.7,
                 label=f"Predictive ({best_model})")
    axes[2].axhline(y=cfg.SLA_LATENCY_LIMIT, color="gray", ls="--", lw=1,
                    label="SLA Limit")
    axes[2].set_ylabel("Normalised Latency")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = cfg.PLOTS_DIR / f"{dataset_name}_scaling_trace.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_sla_bars(scaling_df: pd.DataFrame):
    """Plot 2: SLA violation rate bar chart grouped by policy+model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = scaling_df["dataset"].unique()
    policies = scaling_df.apply(
        lambda r: f"{r['policy']}" if r['model'] == 'none'
                  else f"pred-{r['model']}", axis=1
    )
    scaling_df = scaling_df.copy()
    scaling_df["label"] = policies

    labels = scaling_df["label"].unique()
    x = np.arange(len(datasets))
    width = 0.15

    colors = {"reactive": "#e74c3c", "pred-MA": "#f39c12",
              "pred-LR": "#3498db", "pred-HW": "#2ecc71", "pred-GB": "#9b59b6"}

    for i, label in enumerate(labels):
        vals = []
        for d in datasets:
            row = scaling_df[(scaling_df["dataset"] == d) &
                             (scaling_df["label"] == label)]
            vals.append(row["sla_violation_rate (%)"].values[0] if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=label,
               color=colors.get(label, "gray"))

    ax.set_xticks(x + width * len(labels) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=8)
    ax.set_ylabel("SLA Violation Rate (%)")
    ax.set_title("SLA Violations: Reactive vs Predictive Scaling")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "sla_violation_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_cost_bars(scaling_df: pd.DataFrame):
    """Plot 3: Total cost bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = scaling_df["dataset"].unique()
    scaling_df = scaling_df.copy()
    policies = scaling_df.apply(
        lambda r: f"{r['policy']}" if r['model'] == 'none'
                  else f"pred-{r['model']}", axis=1
    )
    scaling_df["label"] = policies
    labels = scaling_df["label"].unique()
    x = np.arange(len(datasets))
    width = 0.15

    colors = {"reactive": "#e74c3c", "pred-MA": "#f39c12",
              "pred-LR": "#3498db", "pred-HW": "#2ecc71", "pred-GB": "#9b59b6"}

    for i, label in enumerate(labels):
        vals = []
        for d in datasets:
            row = scaling_df[(scaling_df["dataset"] == d) &
                             (scaling_df["label"] == label)]
            vals.append(row["total_cost ($)"].values[0] if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=label,
               color=colors.get(label, "gray"))

    ax.set_xticks(x + width * len(labels) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=8)
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Infrastructure Cost: Reactive vs Predictive Scaling")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = cfg.PLOTS_DIR / "cost_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_error_boxplot():
    """Plot 5: Forecast error distribution (box plot per model)."""
    fig, axes = plt.subplots(1, len(cfg.DATASETS), figsize=(5*len(cfg.DATASETS), 5))
    if len(cfg.DATASETS) == 1:
        axes = [axes]

    for ax, name in zip(axes, cfg.DATASETS):
        path = cfg.PREDICTIONS_DIR / f"{name}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        actual = df["actual"].values

        errors = {}
        for col in df.columns:
            if col.startswith("pred_"):
                model = col.replace("pred_", "")
                errors[model] = df[col].values - actual

        try:
            ax.boxplot(list(errors.values()), tick_labels=list(errors.keys()), showfliers=False)
        except TypeError:
            ax.boxplot(list(errors.values()), labels=list(errors.keys()), showfliers=False)
        ax.axhline(y=0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("Prediction Error")
        ax.set_title(name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Forecast Error Distribution (Test Set)", fontsize=14)
    fig.tight_layout()
    out = cfg.PLOTS_DIR / "error_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("STEP 5: Evaluation metrics and plots")
    print("="*60)

    # A. Forecasting metrics
    fc_table = compute_forecasting_metrics()

    # B. Scaling summary
    sc_table = load_scaling_summary()

    # C. Plots
    print("\n=== Generating Plots ===")
    for name in cfg.DATASETS:
        pred_path = cfg.PREDICTIONS_DIR / f"{name}_predictions.csv"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            plot_forecasts(name, pred_df)
            plot_scaling_trace(name)

    if sc_table is not None:
        plot_sla_bars(sc_table)
        plot_cost_bars(sc_table)

    plot_error_boxplot()

    print(f"\n[OK] All plots saved in {cfg.PLOTS_DIR}/")
    print("[OK] Evaluation complete.")
