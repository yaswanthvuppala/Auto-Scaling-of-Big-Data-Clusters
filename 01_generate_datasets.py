"""
01_generate_datasets.py  –  Create three synthetic cluster-trace datasets.

Each dataset mimics the statistical properties of a real public trace:
  D1: Alibaba Cluster Trace 2018   (batch CPU/memory, diurnal pattern)
  D2: Azure Public Dataset V2 2019 (VM CPU, weekly + diurnal pattern)
  D3: Alibaba GPU Trace 2020       (GPU util, bursty training jobs)

Why synthetic?
  The real traces are 30-100 GB compressed.  We generate data with the
  *same statistical shape* (diurnal peaks, noise, trends) so the
  forecasting + simulation pipeline is fully exercised on realistic
  patterns.  The report should state: "Synthetic data modelled after
  [trace name]; real traces cited in references."

Output:  data/raw/<name>.parquet   (each file ≈ 50 MB in LITE mode,
                                    ≈ 1 GB+ in FULL mode)
"""
import sys, time
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

np.random.seed(cfg.RANDOM_SEED)


def _diurnal_pattern(n_intervals: int, peak_hour: float = 14.0,
                     base: float = 30.0, amplitude: float = 40.0) -> np.ndarray:
    """
    Smooth daily cycle:  low at night, peak around 'peak_hour'.
        pattern(t) = base + amplitude * (1 + cos(2pi(hour - peak_hour)/24)) / 2
    """
    hours = np.arange(n_intervals) * (cfg.INTERVAL_MINUTES / 60.0) % 24
    return base + amplitude * (1 + np.cos(2 * np.pi * (hours - peak_hour) / 24)) / 2


def _weekly_pattern(n_intervals: int) -> np.ndarray:
    """Weekend dip: multiply by 0.7 on Sat/Sun, 1.0 on weekdays."""
    day_idx = np.arange(n_intervals) // cfg.INTERVALS_PER_DAY % 7
    return np.where(day_idx >= 5, 0.7, 1.0)


def generate_dataset(name: str, info: dict):
    """Generate one raw dataset and save as Parquet."""
    n_machines = info["machines"]
    has_gpu    = info["has_gpu"]
    n_intervals = cfg.DAYS * cfg.INTERVALS_PER_DAY  # timestamps per machine

    print(f"\n{'='*60}")
    print(f"Generating {name}:  {n_machines} machines x {n_intervals} intervals")
    print(f"  Total rows = {n_machines * n_intervals:,}")
    t0 = time.time()

    # ── time axis ──────────────────────────────────────────────
    start_ts = pd.Timestamp("2023-01-01")
    timestamps = pd.date_range(start_ts, periods=n_intervals,
                               freq=f"{cfg.INTERVAL_MINUTES}min").astype("datetime64[ms]")

    # ── base workload patterns ─────────────────────────────────
    diurnal = _diurnal_pattern(n_intervals)
    weekly  = _weekly_pattern(n_intervals)
    base    = diurnal * weekly

    # ── generate per-machine data in chunks (memory-friendly) ──
    chunk_size = min(n_machines, 500)
    all_chunks = []

    for chunk_start in range(0, n_machines, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_machines)
        n_in_chunk = chunk_end - chunk_start

        # broadcast timestamps × machines
        n_rows = n_in_chunk * n_intervals
        ts_col = np.tile(timestamps.values, n_in_chunk)
        mids   = np.repeat(
            [f"m_{i:05d}" for i in range(chunk_start, chunk_end)],
            n_intervals
        )

        # CPU: base pattern + per-machine offset + noise + occasional spikes
        offsets = np.random.uniform(-10, 10, n_in_chunk)
        cpu = np.repeat(offsets, n_intervals) + np.tile(base, n_in_chunk)
        cpu += np.random.normal(0, 5, n_rows)                    # Gaussian noise
        spike_mask = np.random.random(n_rows) < 0.005            # 0.5 % spikes
        cpu[spike_mask] += np.random.uniform(20, 40, spike_mask.sum())
        cpu = np.clip(cpu, 0, 100)

        # Memory: correlated with CPU (ρ ≈ 0.6) + own noise
        mem = 0.6 * cpu + 0.4 * np.random.uniform(10, 70, n_rows)
        mem = np.clip(mem, 0, 100)

        # Disk I/O & Network (lighter correlation)
        disk_io = np.abs(np.random.normal(50, 25, n_rows))
        net_io  = np.abs(np.random.normal(30, 20, n_rows))

        data = {
            "timestamp":  ts_col,
            "machine_id": mids,
            "cpu_util":   cpu.astype(np.float32),
            "mem_util":   mem.astype(np.float32),
            "disk_io":    disk_io.astype(np.float32),
            "net_io":     net_io.astype(np.float32),
        }

        if has_gpu:
            # GPU util: bursty — long-running training jobs at high util,
            # then idle.  Modelled as alternating on/off blocks.
            gpu = np.zeros(n_rows, dtype=np.float32)
            for m in range(n_in_chunk):
                s = m * n_intervals
                e = s + n_intervals
                # random training blocks (20-60 intervals long)
                idx = s
                while idx < e:
                    if np.random.random() < 0.4:  # 40 % chance to start a job
                        block_len = np.random.randint(20, 60)
                        end_idx = min(idx + block_len, e)
                        gpu[idx:end_idx] = np.random.uniform(70, 99,
                                                              end_idx - idx)
                        idx = end_idx
                    else:
                        skip = np.random.randint(5, 30)
                        idx += skip
            gpu += np.random.normal(0, 3, n_rows)
            gpu = np.clip(gpu, 0, 100).astype(np.float32)
            data["gpu_util"] = gpu

        all_chunks.append(pd.DataFrame(data))

        pct = (chunk_end / n_machines) * 100
        print(f"  {pct:5.1f}% generated  ({chunk_end}/{n_machines} machines)")

    df = pd.concat(all_chunks, ignore_index=True)

    # ── save as Parquet ────────────────────────────────────────
    out_path = cfg.DATA_RAW_DIR / f"{name}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Saved {out_path.name}: {len(df):,} rows, {size_mb:.1f} MB, "
          f"{elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Generate synthetic cluster-trace datasets")
    print(f"  LITE_MODE = {cfg.LITE_MODE}  "
          f"(set to False in config.py for 1 GB+ files)")
    print("="*60)

    for name, info in cfg.DATASETS.items():
        generate_dataset(name, info)

    print("\n[OK] All datasets generated in data/raw/")
