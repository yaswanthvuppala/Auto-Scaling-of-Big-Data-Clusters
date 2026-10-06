"""
run_all.py  –  Master script: execute the full pipeline end-to-end.

Usage:
    python run_all.py          # run all 5 steps
    python run_all.py 3 5      # run only steps 3 through 5

Each step is idempotent: it reads its inputs and overwrites its outputs,
so you can safely re-run any step after tweaking config.py.
"""
import sys, subprocess, time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


STEPS = [
    ("01_generate_datasets.py",   "Generate synthetic datasets"),
    ("02_spark_preprocess.py",    "PySpark preprocessing + benchmark"),
    ("03_train_forecasters.py",   "Train forecasting models"),
    ("04_cluster_simulator.py",   "Run scaling simulations"),
    ("05_evaluate_and_plot.py",   "Evaluate and generate plots"),
]


def run_step(script: str, description: str):
    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"# Running: python {script}")
    print(f"{'#'*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - t0
    print(f"\n  [OK] {script} completed in {elapsed:.1f}s")
    return result.returncode


if __name__ == "__main__":
    # Parse optional step range from command line
    start_step = 1
    end_step = len(STEPS)
    if len(sys.argv) >= 2:
        start_step = int(sys.argv[1])
    if len(sys.argv) >= 3:
        end_step = int(sys.argv[2])

    print("="*60)
    print("  Predictive Auto-Scaling Pipeline")
    print(f"  Running steps {start_step} -> {end_step}")
    print("="*60)

    t_total = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        if start_step <= i <= end_step:
            run_step(script, desc)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  [OK] Pipeline complete! Total time: {total_elapsed:.1f}s")
    print(f"  Check results/ for metrics and plots.")
    print(f"{'='*60}")
