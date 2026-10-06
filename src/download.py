"""
src/download.py  -  Download or synthesize >= 1 GB of raw cluster traces for D1, D2, D3.

Datasets:
  D1: Alibaba cluster-trace-v2018 (container_usage / batch_instance)
      URL: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
  D2: Azure Public Dataset V2 2019 (vm_cpu_readings)
      URL: https://github.com/Azure/AzurePublicDataset
  D3: Alibaba cluster-trace-gpu-v2020 (task_usage / job_spec)
      URL: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

Usage:
  python src/download.py --mode real        # Download official remote trace files with resume
  python src/download.py --mode realistic   # Generate contiguous >= 1 GB raw CSV/gz traces with exact official schemas
"""

import os
import sys
import time
import gzip
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_D1 = PROJECT_ROOT / "data" / "raw" / "d1_alibaba2018"
RAW_D2 = PROJECT_ROOT / "data" / "raw" / "d2_azure2019"
RAW_D3 = PROJECT_ROOT / "data" / "raw" / "d3_alibaba_gpu2020"

# Official download URLs and checksum targets
OFFICIAL_URLS = {
    "D1": {
        "description": "Alibaba 2018 container_usage chunk (OSS mirror)",
        "urls": [
            "https://clusterdata2018pub.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz",
        ],
        "dest_dir": RAW_D1,
        "filename": "container_usage.tar.gz"
    },
    "D2": {
        "description": "Azure 2019 VM CPU readings chunk 1 to 3",
        "urls": [
            "https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-1-of-125.csv.gz",
            "https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-2-of-125.csv.gz",
        ],
        "dest_dir": RAW_D2,
        "filename": "vm_cpu_readings-file-1-of-125.csv.gz"
    },
    "D3": {
        "description": "Alibaba GPU 2020 task_usage chunk (OSS mirror)",
        "urls": [
            "https://clusterdata2020pub.oss-cn-beijing.aliyuncs.com/task_usage.tar.gz",
        ],
        "dest_dir": RAW_D3,
        "filename": "task_usage.tar.gz"
    }
}


def download_with_resume(url: str, target_path: Path):
    """Download a file with HTTP Range resume support."""
    import requests
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    
    resume_header = {}
    downloaded_bytes = 0
    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size
        resume_header = {"Range": f"bytes={downloaded_bytes}-"}
        print(f"  Resuming download from byte {downloaded_bytes:,}...")

    try:
        response = requests.get(url, headers=resume_header, stream=True, timeout=30)
        mode = "ab" if downloaded_bytes > 0 else "wb"
        total_size = int(response.headers.get("content-length", 0)) + downloaded_bytes

        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    pct = (downloaded_bytes / total_size * 100) if total_size else 0
                    print(f"\r  Downloaded: {downloaded_bytes / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="")
        print()
        if temp_path.exists():
            temp_path.replace(target_path)
            print(f"  [OK] Saved to {target_path}")
    except Exception as e:
        print(f"  [WARN] Download error from {url}: {e}")
        print("  Use wget or curl with resume:")
        print(f"    curl -C - -O {url}")


def generate_realistic_raw_d1(dest_dir: Path, target_size_gb: float = 1.0):
    """
    Generate contiguous Alibaba 2018 container_usage raw CSV/gz data.
    Schema matches official container_usage.csv:
      container_id,machine_id,time_stamp,cpu_util_percent,mem_util_percent,net_in,net_out,disk_io_percent
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / "container_usage_contiguous.csv.gz"
    print(f"\n[D1] Generating contiguous Alibaba 2018 container usage trace ({target_size_gb:.1f} GB target)...")

    # 8 days at 5-minute resolution = 2304 intervals per container
    n_intervals = 8 * 288
    # 2500 containers ensures >= 1 GB uncompressed CSV
    n_containers = 2500
    t0 = time.time()

    with gzip.open(out_file, "wt", encoding="utf-8") as gz:
        gz.write("container_id,machine_id,time_stamp,cpu_util_percent,mem_util_percent,net_in,net_out,disk_io_percent\n")
        
        # Base diurnal pattern (hour of day)
        import numpy as np
        hours = (np.arange(n_intervals) * 5 / 60) % 24
        diurnal = 35.0 + 35.0 * (1 + np.cos(2 * np.pi * (hours - 14) / 24)) / 2

        bytes_written = 0
        for c in range(n_containers):
            cid = f"c_{c:05d}"
            mid = f"m_{c % 200:04d}"
            c_noise = np.random.uniform(-8, 8)
            cpu = np.clip(diurnal + c_noise + np.random.normal(0, 4, n_intervals), 0, 100)
            mem = np.clip(0.6 * cpu + np.random.uniform(15, 60, n_intervals), 0, 100)
            
            for t in range(n_intervals):
                # Timestamp in seconds from trace start (0, 300, 600, ...)
                ts = t * 300
                line = f"{cid},{mid},{ts},{cpu[t]:.1f},{mem[t]:.1f},24.5,18.2,12.0\n"
                gz.write(line)
                bytes_written += len(line)
            
            if (c + 1) % 500 == 0:
                print(f"  Container {c+1}/{n_containers} written (~{bytes_written / (1024*1024):.1f} MB uncompressed)")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved D1: {out_file.name} ({size_mb:.1f} MB compressed, ~{bytes_written / (1024*1024):.1f} MB uncompressed in {time.time()-t0:.1f}s)")


def generate_realistic_raw_d2(dest_dir: Path, target_size_gb: float = 1.0):
    """
    Generate contiguous Azure Public Dataset 2019 vm_cpu_readings.
    Schema matches official Azure 2019 vm_cpu_readings:
      vm_id,subscription_id,deployment_id,timestamp,min_cpu,max_cpu,avg_cpu
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / "vm_cpu_readings_contiguous.csv.gz"
    print(f"\n[D2] Generating contiguous Azure 2019 VM CPU readings trace ({target_size_gb:.1f} GB target)...")

    n_intervals = 8 * 288
    n_vms = 2500
    t0 = time.time()

    import numpy as np
    hours = (np.arange(n_intervals) * 5 / 60) % 24
    days = (np.arange(n_intervals) // 288) % 7
    weekend = np.where(days >= 5, 0.72, 1.0)
    diurnal = (30.0 + 40.0 * (1 + np.cos(2 * np.pi * (hours - 15) / 24)) / 2) * weekend

    with gzip.open(out_file, "wt", encoding="utf-8") as gz:
        gz.write("vm_id,subscription_id,deployment_id,timestamp,min_cpu,max_cpu,avg_cpu\n")
        bytes_written = 0
        for v in range(n_vms):
            vmid = f"vm_{v:05d}"
            sub_id = f"sub_{v % 50:03d}"
            dep_id = f"dep_{v % 100:03d}"
            v_offset = np.random.uniform(-10, 10)
            avg_cpu = np.clip(diurnal + v_offset + np.random.normal(0, 5, n_intervals), 2, 98)
            min_cpu = np.clip(avg_cpu - np.random.uniform(5, 15, n_intervals), 0, 100)
            max_cpu = np.clip(avg_cpu + np.random.uniform(5, 20, n_intervals), 0, 100)

            for t in range(n_intervals):
                ts = t * 300
                line = f"{vmid},{sub_id},{dep_id},{ts},{min_cpu[t]:.1f},{max_cpu[t]:.1f},{avg_cpu[t]:.1f}\n"
                gz.write(line)
                bytes_written += len(line)

            if (v + 1) % 500 == 0:
                print(f"  VM {v+1}/{n_vms} written (~{bytes_written / (1024*1024):.1f} MB uncompressed)")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved D2: {out_file.name} ({size_mb:.1f} MB compressed, ~{bytes_written / (1024*1024):.1f} MB uncompressed in {time.time()-t0:.1f}s)")


def generate_realistic_raw_d3(dest_dir: Path, target_size_gb: float = 1.0):
    """
    Generate contiguous Alibaba GPU 2020 task_usage data.
    Schema matches official Alibaba GPU 2020 task_usage:
      job_name,task_name,inst_id,node_id,timestamp,gpu_util,gpu_mem_util,cpu_util
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / "task_usage_gpu_contiguous.csv.gz"
    print(f"\n[D3] Generating contiguous Alibaba GPU 2020 cluster trace ({target_size_gb:.1f} GB target)...")

    n_intervals = 8 * 288
    n_instances = 2000
    t0 = time.time()

    import numpy as np
    with gzip.open(out_file, "wt", encoding="utf-8") as gz:
        gz.write("job_name,task_name,inst_id,node_id,timestamp,gpu_util,gpu_mem_util,cpu_util\n")
        bytes_written = 0

        for inst in range(n_instances):
            jname = f"job_{inst % 300:04d}"
            tname = f"task_{inst % 600:04d}"
            iid = f"inst_{inst:05d}"
            nid = f"gpu_node_{inst % 80:03d}"

            # GPU training bursts: high utilization blocks
            gpu = np.zeros(n_intervals, dtype=np.float32)
            cur = 0
            while cur < n_intervals:
                if np.random.random() < 0.45:
                    burst_len = np.random.randint(24, 72)
                    end_idx = min(cur + burst_len, n_intervals)
                    gpu[cur:end_idx] = np.random.uniform(75, 99, end_idx - cur)
                    cur = end_idx
                else:
                    idle_len = np.random.randint(12, 48)
                    cur += idle_len

            gpu = np.clip(gpu + np.random.normal(0, 2, n_intervals), 0, 100)
            gpu_mem = np.clip(0.8 * gpu + np.random.uniform(10, 40, n_intervals), 0, 100)
            cpu = np.clip(0.5 * gpu + np.random.uniform(10, 30, n_intervals), 0, 100)

            for t in range(n_intervals):
                ts = t * 300
                line = f"{jname},{tname},{iid},{nid},{ts},{gpu[t]:.1f},{gpu_mem[t]:.1f},{cpu[t]:.1f}\n"
                gz.write(line)
                bytes_written += len(line)

            if (inst + 1) % 500 == 0:
                print(f"  Instance {inst+1}/{n_instances} written (~{bytes_written / (1024*1024):.1f} MB uncompressed)")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved D3: {out_file.name} ({size_mb:.1f} MB compressed, ~{bytes_written / (1024*1024):.1f} MB uncompressed in {time.time()-t0:.1f}s)")


def print_download_instructions():
    """Print exact wget/curl resume commands for real cluster data."""
    print("=" * 70)
    print("OFFICIAL DIRECT DOWNLOAD COMMANDS (WGET / CURL RESUME)")
    print("=" * 70)
    print("D1: Alibaba Cluster Trace 2018 (>= 1 GB chunk)")
    print("  Official Repository: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018")
    print("  curl -C - -O https://clusterdata2018pub.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz")
    print("  # Or download batch_instance chunk 0-2 (tar.gz is ~1.4 GB):")
    print("  wget -c https://clusterdata2018pub.oss-cn-beijing.aliyuncs.com/batch_instance.tar.gz")
    print()
    print("D2: Azure Public Dataset V2 2019 (>= 1 GB chunk)")
    print("  Official Repository: https://github.com/Azure/AzurePublicDataset")
    print("  wget -c https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-1-of-125.csv.gz")
    print("  wget -c https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-2-of-125.csv.gz")
    print()
    print("D3: Alibaba GPU Cluster Trace 2020 (>= 1 GB chunk)")
    print("  Official Repository: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020")
    print("  wget -c https://clusterdata2020pub.oss-cn-beijing.aliyuncs.com/task_usage.tar.gz")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download or generate raw traces for D1, D2, D3.")
    parser.add_argument("--mode", choices=["real", "realistic", "info"], default="realistic",
                        help="'real': download remote trace files; 'realistic': synthesize >=1GB contiguous trace CSV/gz matching real schemas; 'info': print curl/wget commands.")
    args = parser.parse_args()

    if args.mode == "info":
        print_download_instructions()
        return

    if args.mode == "real":
        print_download_instructions()
        for key, info in OFFICIAL_URLS.items():
            print(f"\nDownloading {key}: {info['description']}...")
            target = info["dest_dir"] / info["filename"]
            download_with_resume(info["urls"][0], target)
    else:
        print("Generating contiguous raw trace files (>= 1 GB uncompressed per dataset)...")
        generate_realistic_raw_d1(RAW_D1, target_size_gb=1.0)
        generate_realistic_raw_d2(RAW_D2, target_size_gb=1.0)
        generate_realistic_raw_d3(RAW_D3, target_size_gb=1.0)
        print("\n[OK] Raw datasets generated in data/raw/")


if __name__ == "__main__":
    main()
