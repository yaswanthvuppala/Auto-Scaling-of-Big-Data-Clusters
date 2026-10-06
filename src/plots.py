"""
src/plots.py  -  Generate publication-quality figures (300 DPI) for EDA and experimental evaluation.

Produces:
  A. Exploratory Data Analysis (EDA) Figures (Step 6):
     - Full time-series
     - One zoomed day
     - Load distribution histogram
     - Autocorrelation Function (ACF) plot
     - Daily / hourly diurnal pattern
  B. Experimental Evaluation Figures (Step 15):
     1. Load vs. Capacity trace (Reactive vs. Best Predictive zoomed on peak)
     2. SLA violation rate per policy across datasets
     3. Infrastructure cost per policy across datasets
     4. Cost vs. SLA Pareto trade-off scatter plot
     5. Impact of VM provisioning delay on SLA violation rate
     6. Safety margin effect on Cost vs. SLA compliance
     7. Forecasting error (MAE/RMSE) per predictor
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


# ── Step 6: Exploratory Data Analysis Plots ─────────────────────────────────
def generate_eda_plots():
    print("=" * 70)
    print("STEP 6: GENERATING EXPLORATORY DATA ANALYSIS (EDA) FIGURES")
    print("=" * 70)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for d_name in ["D1", "D2", "D3"]:
        csv_path = PROCESSED_DIR / f"{d_name}.csv"
        if not csv_path.exists():
            print(f"Processed file {csv_path} not found.")
            continue

        df = pd.read_csv(csv_path)
        y = df["load"].values
        ts = pd.to_datetime(df["timestamp"])

        # 1. Full Time-Series + Zoomed Day
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        axes[0].plot(ts, y, color="#2980b9", lw=1.2, label=f"{d_name} Load")
        axes[0].set_ylabel("Workload Demand", fontsize=10)
        axes[0].set_title(f"Full Workload Time-Series ({d_name})", fontsize=12, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(loc="upper right")

        # Zoomed single day (first 288 points)
        zoom_ts = ts.iloc[:288]
        zoom_y = y[:288]
        axes[1].plot(zoom_ts, zoom_y, color="#e67e22", lw=1.5, marker=".", markersize=3, label="Zoomed 24-Hour Day")
        axes[1].set_xlabel("Time (5-Minute Intervals)", fontsize=10)
        axes[1].set_ylabel("Workload Demand", fontsize=10)
        axes[1].set_title(f"{d_name} - Diurnal 24-Hour Zoom Window", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(loc="upper right")

        fig.tight_layout()
        out_ts = FIGURES_DIR / f"{d_name}_timeseries_and_zoom.png"
        fig.savefig(out_ts, dpi=300)
        plt.close(fig)
        print(f"  [OK] Saved {out_ts.name}")

        # 2. Histogram & Autocorrelation (ACF)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(y, bins=35, color="#27ae60", edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(y), color="red", linestyle="--", lw=1.5, label=f"Mean: {np.mean(y):.1f}")
        axes[0].set_xlabel("Workload Demand Units", fontsize=10)
        axes[0].set_ylabel("Frequency Count", fontsize=10)
        axes[0].set_title(f"{d_name} Load Distribution", fontsize=11, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Autocorrelation up to 2 days (576 lags)
        max_lag = min(576, len(y) // 2)
        lags = np.arange(1, max_lag + 1)
        y_mean = np.mean(y)
        y_var = np.var(y)
        acf_vals = [np.mean((y[:-lag] - y_mean) * (y[lag:] - y_mean)) / y_var for lag in lags]

        axes[1].plot(lags, acf_vals, color="#8e44ad", lw=1.5)
        axes[1].axhline(0, color="gray", linestyle="--", lw=0.8)
        axes[1].axvline(288, color="red", linestyle=":", lw=1.2, label="24-Hour Lag (288)")
        axes[1].set_xlabel("Lag (5-Minute Steps)", fontsize=10)
        axes[1].set_ylabel("Autocorrelation", fontsize=10)
        axes[1].set_title(f"{d_name} Autocorrelation Function (ACF)", fontsize=11, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        out_hist = FIGURES_DIR / f"{d_name}_histogram_and_acf.png"
        fig.savefig(out_hist, dpi=300)
        plt.close(fig)
        print(f"  [OK] Saved {out_hist.name}")

        # 3. Hourly Diurnal Average Pattern
        fig, ax = plt.subplots(figsize=(8, 4))
        df["hour"] = ts.dt.hour
        hourly_stats = df.groupby("hour")["load"].agg(["mean", "std"])
        ax.plot(hourly_stats.index, hourly_stats["mean"], color="#d35400", lw=2, marker="o", label="Hourly Mean Demand")
        ax.fill_between(hourly_stats.index,
                        hourly_stats["mean"] - hourly_stats["std"],
                        hourly_stats["mean"] + hourly_stats["std"],
                        color="#d35400", alpha=0.2, label="+/- 1 Std Dev")
        ax.set_xlabel("Hour of Day (0 to 23)", fontsize=10)
        ax.set_ylabel("Average Workload", fontsize=10)
        ax.set_title(f"{d_name} - Diurnal Hourly Profile", fontsize=11, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        out_diurnal = FIGURES_DIR / f"{d_name}_hourly_diurnal_pattern.png"
        fig.savefig(out_diurnal, dpi=300)
        plt.close(fig)
        print(f"  [OK] Saved {out_diurnal.name}")


# ── Step 15: Experimental Evaluation Plots ──────────────────────────────────
def generate_experiment_plots(results_csv: Optional[Path] = None):
    print("\n" + "=" * 70)
    print("STEP 15: GENERATING REPORT-QUALITY EXPERIMENT FIGURES (300 DPI)")
    print("=" * 70)
    if results_csv is None:
        results_csv = TABLES_DIR / "all_results.csv"

    if not results_csv.exists():
        print(f"Results file {results_csv} not found. Run `src/run_experiments.py` first.")
        return

    df = pd.read_csv(results_csv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SLA Violation Rate per Policy
    fig, ax = plt.subplots(figsize=(10, 5))
    policy_order = df["policy"].unique()
    datasets = df["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(policy_order)

    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#34495e"]
    for i, p in enumerate(policy_order):
        sub = df[df["policy"] == p].groupby("dataset")["sla_violation_rate_pct"].mean()
        vals = [sub.get(d, 0.0) for d in datasets]
        ax.bar(x + i * width, vals, width, label=p, color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * (len(policy_order) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("SLA Violation Rate (%)", fontsize=11)
    ax.set_title("SLA Violation Rate by Auto-Scaling Policy Across Datasets", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out1 = FIGURES_DIR / "fig2_sla_violation_by_policy.png"
    fig.savefig(out1, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {out1.name}")

    # 2. Total Cost per Policy
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(policy_order):
        sub = df[df["policy"] == p].groupby("dataset")["total_cost_usd"].mean()
        vals = [sub.get(d, 0.0) for d in datasets]
        ax.bar(x + i * width, vals, width, label=p, color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * (len(policy_order) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Total Infrastructure Cost ($)", fontsize=11)
    ax.set_title("Total Infrastructure Cost by Scaling Policy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out2 = FIGURES_DIR / "fig3_cost_by_policy.png"
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {out2.name}")

    # 3. Cost vs. SLA Pareto Scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, p in enumerate(policy_order):
        sub = df[df["policy"] == p]
        ax.scatter(sub["total_cost_usd"], sub["sla_violation_rate_pct"],
                   s=100, label=p, color=colors[i % len(colors)], edgecolors="black", alpha=0.85)
    ax.set_xlabel("Total Cost ($)", fontsize=11)
    ax.set_ylabel("SLA Violation Rate (%)", fontsize=11)
    ax.set_title("Cost vs. SLA Violation Pareto Frontier", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Policy", fontsize=9)
    fig.tight_layout()
    out3 = FIGURES_DIR / "fig4_pareto_cost_vs_sla.png"
    fig.savefig(out3, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {out3.name}")

    # 4. Impact of Provisioning Delay on SLA
    if "provisioning_delay" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        delays = sorted(df["provisioning_delay"].unique())
        for p in ["Reactive", "Predictive-GradientBoosting", "Hybrid-GradientBoosting"]:
            if p in df["policy"].values:
                sub = df[df["policy"] == p].groupby("provisioning_delay")["sla_violation_rate_pct"].mean()
                ax.plot(delays, [sub.get(d, 0.0) for d in delays], marker="o", lw=2, label=p)
        ax.set_xlabel("VM Provisioning Delay (Steps = 5-Min Multiples)", fontsize=11)
        ax.set_ylabel("Mean SLA Violation Rate (%)", fontsize=11)
        ax.set_title("Impact of VM Provisioning Delay on SLA Violations", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out4 = FIGURES_DIR / "fig5_provisioning_delay_vs_sla.png"
        fig.savefig(out4, dpi=300)
        plt.close(fig)
        print(f"  [OK] Saved {out4.name}")

    # 5. Safety Margin Effect on Cost & SLA
    if "safety_margin" in df.columns:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        margins = sorted(df["safety_margin"].unique())
        sub_p = df[df["policy"].str.startswith("Predictive")].groupby("safety_margin")
        cost_means = [sub_p.get_group(m)["total_cost_usd"].mean() for m in margins if m in sub_p.groups]
        sla_means  = [sub_p.get_group(m)["sla_violation_rate_pct"].mean() for m in margins if m in sub_p.groups]

        color1 = "#2980b9"
        ax1.set_xlabel("Safety Margin Overhead (0%, 10%, 20%)", fontsize=11)
        ax1.set_ylabel("Total Cost ($)", color=color1, fontsize=11)
        ax1.plot([f"{int(m*100)}%" for m in margins], cost_means, color=color1, marker="s", lw=2, label="Cost ($)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#e74c3c"
        ax2.set_ylabel("SLA Violation Rate (%)", color=color2, fontsize=11)
        ax2.plot([f"{int(m*100)}%" for m in margins], sla_means, color=color2, marker="^", lw=2, linestyle="--", label="SLA Viol. (%)")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_title("Trade-off: Safety Margin vs. Cost and SLA Compliance", fontsize=12, fontweight="bold")
        fig.tight_layout()
        out5 = FIGURES_DIR / "fig6_safety_margin_tradeoff.png"
        fig.savefig(out5, dpi=300)
        plt.close(fig)
        print(f"  [OK] Saved {out5.name}")


if __name__ == "__main__":
    generate_eda_plots()
    generate_experiment_plots()
