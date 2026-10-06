"""
02_spark_preprocess.py  –  PySpark (local mode) preprocessing + node benchmark.

What this script does:
  1. Reads each raw Parquet file (millions of rows, per-machine data).
  2. Aggregates per-machine readings into CLUSTER-LEVEL metrics at each
     5-minute timestamp:
         cpu_utilization  = mean(cpu_util)   across all machines
         mem_utilization  = mean(mem_util)   across all machines
         total_cpu_demand = sum(cpu_util)     (used by the simulator)
  3. Saves the result as a compact Parquet (~8 640 rows = 30 days × 288/day).
  4. Runs a Spark benchmark to estimate per-node processing throughput.
     This measurement is written to models/spark_benchmark.json and fed
     into the simulator as the NODE_CAPACITY parameter.

Why PySpark instead of plain pandas?
  - Demonstrates big-data processing on a realistic pipeline.
  - Handles 1 GB+ files without loading everything into RAM at once.
  - Local mode still uses Spark's optimiser (Catalyst), partitioning, and
    lazy evaluation — the same engine that runs on a 1000-node cluster.
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure PySpark uses the running Python interpreter on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import config as cfg

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark():
    """Create a local-mode Spark session."""
    return (
        SparkSession.builder
        .master("local[*]")                          # use all CPU cores
        .appName("PredictiveAutoScale-Preprocess")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def preprocess_dataset(spark, name: str, info: dict):
    """
    Read raw Parquet → aggregate to cluster-level time series → save.

    Aggregation logic:
        GROUP BY timestamp
        cpu_utilization  = AVG(cpu_util)      (mean across machines)
        mem_utilization  = AVG(mem_util)
        total_cpu_demand = SUM(cpu_util)       (total demand in CPU-units)
        machine_count    = COUNT(DISTINCT machine_id)
    """
    raw_path = str(cfg.DATA_RAW_DIR / f"{name}.parquet")
    print(f"\n  Reading {raw_path} ...")
    df = spark.read.parquet(raw_path)
    row_count = df.count()
    print(f"    Raw rows: {row_count:,}")

    # ── aggregate ──────────────────────────────────────────────
    agg_df = (
        df.groupBy("timestamp")
        .agg(
            F.avg("cpu_util").alias("cpu_utilization"),
            F.avg("mem_util").alias("mem_utilization"),
            F.sum("cpu_util").alias("total_cpu_demand"),
            F.countDistinct("machine_id").alias("machine_count"),
        )
        .orderBy("timestamp")
    )

    # GPU column (only for D3)
    if info["has_gpu"]:
        gpu_agg = (
            df.groupBy("timestamp")
            .agg(F.avg("gpu_util").alias("gpu_utilization"))
        )
        agg_df = agg_df.join(gpu_agg, on="timestamp", how="left")

    # ── save as Parquet ────────────────────────────────────────
    # Collect to pandas (only ~8640 rows) and save with pyarrow
    pdf = agg_df.toPandas()
    pdf = pdf.sort_values("timestamp").reset_index(drop=True)

    out_path = cfg.DATA_PROCESSED_DIR / f"{name}_processed.parquet"
    pdf.to_parquet(out_path, engine="pyarrow")

    print(f"    Aggregated rows: {len(pdf):,}")
    print(f"    Saved -> {out_path.name}")
    return row_count


def run_spark_benchmark(spark, name: str):
    """
    Benchmark: measure how many rows per second ONE Spark core can process.

    This number becomes the NODE_CAPACITY parameter in the simulator:
      "One cluster node can handle X CPU-units of workload per 5-min step."

    Approach:
      1. Read the first raw dataset.
      2. Run a heavy aggregation (simulates real ETL).
      3. Measure wall-clock time.
      4. Derive throughput = total_rows / seconds / num_cores.
    """
    raw_path = str(cfg.DATA_RAW_DIR / f"{name}.parquet")
    df = spark.read.parquet(raw_path)
    df.cache()
    _ = df.count()  # materialise cache

    # heavy aggregation (GROUP BY + window function)
    t0 = time.time()
    result = (
        df.groupBy("machine_id")
        .agg(
            F.avg("cpu_util"),
            F.stddev("cpu_util"),
            F.max("mem_util"),
            F.min("cpu_util"),
        )
        .collect()
    )
    elapsed = time.time() - t0
    n_rows = df.count()

    import os
    n_cores = os.cpu_count() or 4
    throughput_per_core = n_rows / elapsed / n_cores

    # Scale: 1 "node" in our simulator handles throughput_per_core CPU-units
    # We normalise so that the demand values (total_cpu_demand) fit the
    # simulator's capacity model.
    benchmark = {
        "rows_processed":    n_rows,
        "elapsed_seconds":   round(elapsed, 2),
        "cpu_cores_used":    n_cores,
        "throughput_per_core_per_sec": round(throughput_per_core, 1),
        "node_capacity_cpu_units":     round(throughput_per_core / 100, 2),
    }
    out = cfg.MODELS_DIR / "spark_benchmark.json"
    with open(out, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\n  Spark Benchmark:")
    print(f"    {n_rows:,} rows in {elapsed:.2f}s on {n_cores} cores")
    print(f"    Throughput/core: {throughput_per_core:,.0f} rows/s")
    print(f"    Saved -> {out.name}")
    return benchmark


# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("STEP 2: PySpark preprocessing + node-capacity benchmark")
    print("="*60)

    spark = get_spark()
    try:
        first_name = None
        for name, info in cfg.DATASETS.items():
            preprocess_dataset(spark, name, info)
            if first_name is None:
                first_name = name

        run_spark_benchmark(spark, first_name)
    finally:
        spark.stop()

    print("\n[OK] All datasets preprocessed in data/processed/")
    print("[OK] Spark benchmark saved in models/spark_benchmark.json")
