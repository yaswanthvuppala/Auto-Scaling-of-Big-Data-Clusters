"""
src/inspect_data.py  -  Inspect raw cluster traces with explicit PySpark schemas (no inferSchema).

Prints:
  1. File listing and sample records
  2. Row count, minimum and maximum timestamps, and time duration
  3. Null counts across all columns
  4. Workload signal statistics (min, mean, max, stddev)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure PySpark uses running Python interpreter and local Hadoop binaries
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if "HADOOP_HOME" not in os.environ and (PROJECT_ROOT / "hadoop").exists():
    os.environ["HADOOP_HOME"] = str(PROJECT_ROOT / "hadoop")
    os.environ["PATH"] = str(PROJECT_ROOT / "hadoop" / "bin") + ";" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql import functions as F


# ── Explicit Schemas (Never use inferSchema on gigabyte-scale files) ─────────
SCHEMA_D1 = StructType([
    StructField("container_id", StringType(), False),
    StructField("machine_id", StringType(), False),
    StructField("time_stamp", LongType(), False),
    StructField("cpu_util_percent", DoubleType(), True),
    StructField("mem_util_percent", DoubleType(), True),
    StructField("net_in", DoubleType(), True),
    StructField("net_out", DoubleType(), True),
    StructField("disk_io_percent", DoubleType(), True),
])

SCHEMA_D2 = StructType([
    StructField("vm_id", StringType(), False),
    StructField("subscription_id", StringType(), False),
    StructField("deployment_id", StringType(), False),
    StructField("timestamp", LongType(), False),
    StructField("min_cpu", DoubleType(), True),
    StructField("max_cpu", DoubleType(), True),
    StructField("avg_cpu", DoubleType(), True),
])

SCHEMA_D3 = StructType([
    StructField("job_name", StringType(), False),
    StructField("task_name", StringType(), False),
    StructField("inst_id", StringType(), False),
    StructField("node_id", StringType(), False),
    StructField("timestamp", LongType(), False),
    StructField("gpu_util", DoubleType(), True),
    StructField("gpu_mem_util", DoubleType(), True),
    StructField("cpu_util", DoubleType(), True),
])


def get_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("TraceInspection")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def inspect_dataset(spark, name: str, data_dir: Path, schema: StructType, ts_col: str, workload_col: str):
    print("\n" + "=" * 70)
    print(f"INSPECTING DATASET: {name}")
    print(f"Directory: {data_dir}")
    print("=" * 70)

    files = list(data_dir.glob("*.csv*")) + list(data_dir.glob("*.parquet*"))
    if not files:
        print(f"  [WARN] No data files found in {data_dir}. Run `python src/download.py` first.")
        return

    print("Found files:")
    total_size = 0
    for f in files:
        sz_mb = f.stat().st_size / (1024 * 1024)
        total_size += f.stat().st_size
        print(f"  - {f.name} ({sz_mb:.1f} MB)")
    print(f"Total raw size on disk: {total_size / (1024*1024):.1f} MB ({total_size / (1024**3):.2f} GB)")

    # Load with explicit schema
    df = spark.read.schema(schema).option("header", "true").csv(str(data_dir))

    print("\n--- Explicit Schema ---")
    df.printSchema()

    print("\n--- First 5 Sample Rows ---")
    df.show(5, truncate=False)

    print("Computing metrics via Spark distributed execution (no pandas OOM risk)...")
    total_rows = df.count()
    print(f"Total Row Count: {total_rows:,}")

    # Min / Max timestamps
    ts_stats = df.select(
        F.min(ts_col).alias("min_ts"),
        F.max(ts_col).alias("max_ts")
    ).collect()[0]
    min_ts, max_ts = ts_stats["min_ts"], ts_stats["max_ts"]
    duration_hrs = (max_ts - min_ts) / 3600.0 if (min_ts is not None and max_ts is not None) else 0
    print(f"Timestamp range: {min_ts} -> {max_ts} seconds ({duration_hrs:.1f} hours / {duration_hrs/24:.1f} days)")

    # Null counts per column
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    print("\n--- Null Counts Per Column ---")
    for col_name, nulls in null_counts.items():
        pct = (nulls / total_rows * 100) if total_rows > 0 else 0
        print(f"  {col_name:18s}: {nulls:6d} nulls ({pct:4.2f}%)")

    # Workload signal summary stats
    print(f"\n--- Primary Workload Signal Stats ({workload_col}) ---")
    df.select(workload_col).describe().show()


def main():
    spark = get_spark_session()
    try:
        inspect_dataset(spark, "D1 (Alibaba Cluster 2018)",
                        PROJECT_ROOT / "data" / "raw" / "d1_alibaba2018",
                        SCHEMA_D1, "time_stamp", "cpu_util_percent")

        inspect_dataset(spark, "D2 (Azure Public Dataset 2019)",
                        PROJECT_ROOT / "data" / "raw" / "d2_azure2019",
                        SCHEMA_D2, "timestamp", "avg_cpu")

        inspect_dataset(spark, "D3 (Alibaba GPU Cluster 2020)",
                        PROJECT_ROOT / "data" / "raw" / "d3_alibaba_gpu2020",
                        SCHEMA_D3, "timestamp", "gpu_util")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
