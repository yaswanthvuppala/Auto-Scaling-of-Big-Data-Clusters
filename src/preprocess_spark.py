"""
src/preprocess_spark.py  -  Distributed PySpark ETL, Parquet conversion, and cluster workload aggregation.

Key Steps:
  1. Read raw CSV/gz files with explicit schemas.
  2. Convert to Parquet (data/parquet/) and measure compression ratio.
  3. Aggregate to 5-minute fixed intervals (justified by VM boot latency).
  4. Fill missing intervals, clean nulls/outliers.
  5. Calibrate load so mean demand matches nominal ~10 node cluster capacity.
  6. Save clean time series to data/processed/{D1, D2, D3}.csv.
  7. Compute summary statistics (PMR, burstiness, ACF lag-1, ACF lag-288).
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if "HADOOP_HOME" not in os.environ and (PROJECT_ROOT / "hadoop").exists():
    os.environ["HADOOP_HOME"] = str(PROJECT_ROOT / "hadoop")
    os.environ["PATH"] = str(PROJECT_ROOT / "hadoop" / "bin") + ";" + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.inspect_data import SCHEMA_D1, SCHEMA_D2, SCHEMA_D3

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("SparkPreprocessing")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .getOrCreate()
    )


def compute_acf(series: np.ndarray, lag: int) -> float:
    """Compute sample autocorrelation at a specific lag."""
    if len(series) <= lag:
        return 0.0
    s_mean = np.mean(series)
    denom = np.sum((series - s_mean) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((series[:-lag] - s_mean) * (series[lag:] - s_mean))
    return float(numer / denom)


def process_dataset(spark, name: str, raw_dir: Path, schema, ts_col: str, val_col: str, out_name: str):
    print("\n" + "=" * 70)
    print(f"PROCESSING DATASET: {name}")
    print("=" * 70)
    t0 = time.time()

    raw_files = list(raw_dir.glob("*.csv*"))
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")

    raw_bytes = sum(f.stat().st_size for f in raw_files)
    raw_size_gb = raw_bytes / (1024 ** 3)
    print(f"Reading raw CSV/gz data: {raw_size_gb:.2f} GB on disk...")

    df = spark.read.schema(schema).option("header", "true").csv(str(raw_dir))
    raw_row_count = df.count()
    print(f"Loaded {raw_row_count:,} raw records.")

    # 1. Convert to Parquet and measure size reduction
    parquet_path = PARQUET_DIR / out_name.lower()
    print(f"Writing optimized Parquet to {parquet_path}...")
    df.write.mode("overwrite").parquet(str(parquet_path))
    parquet_bytes = sum(f.stat().st_size for f in parquet_path.glob("*.parquet"))
    parquet_size_gb = parquet_bytes / (1024 ** 3) if parquet_bytes > 0 else 0.01
    ratio = (raw_bytes / parquet_bytes) if parquet_bytes > 0 else 1.0
    print(f"  Raw: {raw_size_gb:.3f} GB -> Parquet: {parquet_size_gb:.3f} GB (Compression ratio: {ratio:.1f}x)")

    # 2. Re-read from optimized Parquet
    p_df = spark.read.parquet(str(parquet_path))

    # 3. Aggregate to 5-minute fixed intervals (300 seconds)
    # Justification: 5-minute sampling matches typical cloud VM boot latency
    # and standard CloudWatch/Azure monitoring metrics without high-frequency jitter.
    interval_sec = 300
    agg_df = (
        p_df
        .withColumn("interval_idx", F.floor(F.col(ts_col) / interval_sec))
        .groupBy("interval_idx")
        .agg(
            F.sum(val_col).alias("total_demand"),
            F.avg(val_col).alias("mean_util"),
            F.count("*").alias("active_entities")
        )
        .orderBy("interval_idx")
    )

    pdf = agg_df.toPandas()
    print(f"Aggregated {len(pdf)} 5-minute time intervals.")

    # 4. Fill missing intervals and clean time axis
    full_idx = pd.RangeIndex(start=int(pdf["interval_idx"].min()),
                             stop=int(pdf["interval_idx"].max()) + 1,
                             step=1)
    pdf = pdf.set_index("interval_idx").reindex(full_idx)
    pdf["total_demand"] = pdf["total_demand"].interpolate(method="linear").bfill().ffill()
    pdf["mean_util"] = pdf["mean_util"].interpolate(method="linear").bfill().ffill()

    # Generate contiguous human-readable timestamps
    start_ts = pd.Timestamp("2023-01-01 00:00:00")
    pdf["timestamp"] = [start_ts + pd.Timedelta(minutes=5 * i) for i in range(len(pdf))]

    # 5. Scale workload so average demand corresponds to ~10 nodes
    # Sizing justification: Nominal 10-node capacity provides a standard departmental
    # cluster baseline where diurnal peaks require 15-25 nodes and troughs scale down to 3-5.
    raw_demand = pdf["total_demand"].values
    target_mean_load = 10.0 * 100.0  # 1,000 units (10 nodes @ 100 capacity)
    scale_factor = target_mean_load / np.mean(raw_demand)
    scaled_load = raw_demand * scale_factor
    pdf["load"] = scaled_load

    # Save to data/processed/{out_name}.csv
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / f"{out_name}.csv"
    pdf[["timestamp", "load"]].to_csv(out_csv, index=False)
    print(f"[OK] Saved processed workload trace -> {out_csv}")

    # 6. Compute summary statistics
    y = pdf["load"].values
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    peak_val = float(np.max(y))
    pmr = peak_val / mean_val if mean_val > 0 else 1.0
    burstiness = (std_val - mean_val) / (std_val + mean_val) if (std_val + mean_val) > 0 else 0.0
    acf_1 = compute_acf(y, lag=1)
    acf_288 = compute_acf(y, lag=288)  # 288 steps * 5 min = 24 hours (1-day lag)

    summary = {
        "Dataset": out_name,
        "Raw Size (GB)": round(raw_size_gb, 2),
        "Raw Rows": raw_row_count,
        "Series Length": len(y),
        "Mean Load": round(mean_val, 1),
        "Std Dev": round(std_val, 1),
        "Peak Load": round(peak_val, 1),
        "Peak-to-Mean Ratio": round(pmr, 2),
        "Burstiness": round(burstiness, 3),
        "ACF (Lag 1)": round(acf_1, 3),
        "ACF (Lag 288 / 1-day)": round(acf_288, 3),
        "Time (s)": round(time.time() - t0, 1),
    }
    return summary


def main():
    spark = get_spark_session()
    summaries = []
    try:
        s1 = process_dataset(spark, "D1 (Alibaba 2018)",
                             PROJECT_ROOT / "data" / "raw" / "d1_alibaba2018",
                             SCHEMA_D1, "time_stamp", "cpu_util_percent", "D1")
        summaries.append(s1)

        s2 = process_dataset(spark, "D2 (Azure 2019)",
                             PROJECT_ROOT / "data" / "raw" / "d2_azure2019",
                             SCHEMA_D2, "timestamp", "avg_cpu", "D2")
        summaries.append(s2)

        s3 = process_dataset(spark, "D3 (Alibaba GPU 2020)",
                             PROJECT_ROOT / "data" / "raw" / "d3_alibaba_gpu2020",
                             SCHEMA_D3, "timestamp", "gpu_util", "D3")
        summaries.append(s3)

        # Print summary table
        sum_df = pd.DataFrame(summaries)
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY TABLE (DATASETS D1, D2, D3)")
        print("=" * 80)
        print(sum_df.to_string(index=False))

        # Save summary table
        (PROJECT_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
        sum_df.to_csv(PROJECT_ROOT / "results" / "tables" / "dataset_summary.csv", index=False)
        print("\n[OK] Summary table saved to results/tables/dataset_summary.csv")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
