#!/usr/bin/env bash
# ==============================================================================
# download_and_run_linux.sh
# End-to-end workload trace pipeline for Linux:
#   1. Download raw traces (>= 1 GB each) for D1, D2, D3 with resume support
#   2. Inspect datasets using PySpark explicit schemas (no inferSchema)
#   3. Preprocess & aggregate to 5-min intervals, save to Parquet & processed CSV
#   4. Run PySpark node-capacity benchmark across CPU cores
#   5. Train forecasting models (Naive, MA, LR, Holt-Winters, LightGBM/GB)
#   6. Run cluster auto-scaling simulation across experimental grid
#   7. Generate all 300 DPI publication plots and summary tables
# ==============================================================================

set -euo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BOLD}${BLUE}==============================================================================${NC}"
echo -e "${BOLD}${BLUE}   PREDICTIVE AUTO-SCALING RESEARCH PIPELINE (LINUX RUNNER)                   ${NC}"
echo -e "${BOLD}${BLUE}==============================================================================${NC}"

# ------------------------------------------------------------------------------
# 1. Environment & Dependency Checks
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[1/7] Checking environment & dependencies...${NC}"

if [ -d ".venv" ]; then
    echo -e "  Activating virtual environment: ${GREEN}.venv${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "  Activating virtual environment: ${GREEN}venv${NC}"
    source venv/bin/activate
else
    echo -e "  ${YELLOW}Notice: No active .venv found in project root. Using system Python.${NC}"
fi

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "  ${RED}Error: Python is not installed. Please install Python 3.10/3.11.${NC}"
    exit 1
fi
PYTHON_BIN=$(command -v python3 || command -v python)
echo -e "  Python: ${GREEN}$($PYTHON_BIN --version)${NC}"

# Check Java
if ! command -v java &>/dev/null; then
    echo -e "  ${YELLOW}Warning: Java runtime not detected in PATH. PySpark requires JDK 11 or 17.${NC}"
    echo -e "  Install on Ubuntu/Debian: ${BOLD}sudo apt update && sudo apt install -y openjdk-17-jdk${NC}"
else
    JAVA_VER=$(java -version 2>&1 | head -n 1)
    echo -e "  Java:   ${GREEN}${JAVA_VER}${NC}"
fi

# Ensure output directories exist
mkdir -p data/raw/d1_alibaba2018 \
         data/raw/d2_azure2019 \
         data/raw/d3_alibaba_gpu2020 \
         data/parquet \
         data/processed \
         models \
         results/figures \
         results/tables \
         results/predictions \
         results/simulation

# ------------------------------------------------------------------------------
# 2. Download or Prepare Raw Traces
# ------------------------------------------------------------------------------
DOWNLOAD_MODE="real" # Default: download official trace chunks
if [ "${1:-}" == "--synth" ] || [ "${1:-}" == "--offline" ]; then
    DOWNLOAD_MODE="synth"
fi

echo -e "\n${BOLD}[2/7] Preparing raw cluster trace datasets (>= 1 GB each)...${NC}"

D1_DIR="data/raw/d1_alibaba2018"
D2_DIR="data/raw/d2_azure2019"
D3_DIR="data/raw/d3_alibaba_gpu2020"

has_raw_data() {
    [ -n "$(find "$1" -type f \( -name "*.csv*" -o -name "*.tar.gz" \) 2>/dev/null | head -n 1)" ]
}

if has_raw_data "$D1_DIR" && has_raw_data "$D2_DIR" && has_raw_data "$D3_DIR" && [ "${1:-}" != "--force-download" ]; then
    echo -e "  ${GREEN}Raw datasets already detected in data/raw/. Skipping download.${NC}"
    echo -e "  (Use ${BOLD}./download_and_run_linux.sh --force-download${NC} to force re-download)."
else
    if [ "$DOWNLOAD_MODE" == "real" ]; then
        echo -e "  Downloading official trace chunks via wget/curl with HTTP Range resume..."

        # D1: Alibaba 2018
        echo -e "\n  ${BLUE}--> [D1] Alibaba Cluster Trace 2018 (container_usage)...${NC}"
        mkdir -p "$D1_DIR"
        if ! has_raw_data "$D1_DIR"; then
            if command -v wget &>/dev/null; then
                wget -c -P "$D1_DIR" https://clusterdata2018pub.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz || true
            else
                curl -C - -o "$D1_DIR/container_usage.tar.gz" https://clusterdata2018pub.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz || true
            fi
            if [ -f "$D1_DIR/container_usage.tar.gz" ]; then
                echo "  Extracting $D1_DIR/container_usage.tar.gz..."
                tar -xzf "$D1_DIR/container_usage.tar.gz" -C "$D1_DIR" || true
            fi
        fi

        # D2: Azure Public Dataset 2019
        echo -e "\n  ${BLUE}--> [D2] Azure Public Dataset V2 2019 (vm_cpu_readings)...${NC}"
        mkdir -p "$D2_DIR"
        if ! has_raw_data "$D2_DIR"; then
            if command -v wget &>/dev/null; then
                wget -c -P "$D2_DIR" https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-1-of-125.csv.gz || true
                wget -c -P "$D2_DIR" https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-2-of-125.csv.gz || true
            else
                curl -C - -o "$D2_DIR/vm_cpu_readings-file-1-of-125.csv.gz" https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-1-of-125.csv.gz || true
                curl -C - -o "$D2_DIR/vm_cpu_readings-file-2-of-125.csv.gz" https://azurepublicdataset.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/vm_cpu_readings-file-2-of-125.csv.gz || true
            fi
        fi

        # D3: Alibaba GPU 2020
        echo -e "\n  ${BLUE}--> [D3] Alibaba GPU Cluster Trace 2020 (task_usage)...${NC}"
        mkdir -p "$D3_DIR"
        if ! has_raw_data "$D3_DIR"; then
            if command -v wget &>/dev/null; then
                wget -c -P "$D3_DIR" https://clusterdata2020pub.oss-cn-beijing.aliyuncs.com/task_usage.tar.gz || true
            else
                curl -C - -o "$D3_DIR/task_usage.tar.gz" https://clusterdata2020pub.oss-cn-beijing.aliyuncs.com/task_usage.tar.gz || true
            fi
            if [ -f "$D3_DIR/task_usage.tar.gz" ]; then
                echo "  Extracting $D3_DIR/task_usage.tar.gz..."
                tar -xzf "$D3_DIR/task_usage.tar.gz" -C "$D3_DIR" || true
            fi
        fi

        # Fallback check: if remote network mirrors failed or were blocked, generate contiguous traces
        if ! has_raw_data "$D1_DIR" || ! has_raw_data "$D2_DIR" || ! has_raw_data "$D3_DIR"; then
            echo -e "\n  ${YELLOW}[Notice] Remote mirrors unavailable or network interrupted.${NC}"
            echo -e "  Generating contiguous raw traces (>= 1 GB each) matching exact official schemas..."
            $PYTHON_BIN src/download.py --mode realistic
        fi
    else
        echo -e "  Synthesizing contiguous raw traces (>= 1 GB each) matching official schemas..."
        $PYTHON_BIN src/download.py --mode realistic
    fi
fi

# ------------------------------------------------------------------------------
# 3. Inspect Raw Datasets with PySpark Explicit Schemas (Step 4)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/7] Inspecting raw datasets with PySpark (no inferSchema)...${NC}"
$PYTHON_BIN src/inspect_data.py

# ------------------------------------------------------------------------------
# 4. PySpark Distributed Preprocessing & Parquet Conversion (Step 5)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/7] Running PySpark preprocessing & 5-minute workload aggregation...${NC}"
$PYTHON_BIN src/preprocess_spark.py

# ------------------------------------------------------------------------------
# 5. Measure Real PySpark Scaling Throughput (Step 8)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/7] Calibrating cluster node capacity with PySpark benchmark...${NC}"
$PYTHON_BIN src/benchmark_spark.py

# ------------------------------------------------------------------------------
# 6. Train Models & Run Cluster Simulation Grid (Steps 11-14)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[6/7] Training forecasters and running auto-scaling simulations across D1, D2, D3...${NC}"
$PYTHON_BIN src/run_experiments.py

# ------------------------------------------------------------------------------
# 7. Generate Publication-Quality Plots (Steps 6 & 15)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[7/7] Generating publication-quality figures (300 DPI) and tables...${NC}"
$PYTHON_BIN src/plots.py

echo -e "\n${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}   PIPELINE COMPLETED SUCCESSFULLY!                                           ${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e "Results Summary:"
echo -e "  - Tables:  ${BLUE}results/tables/all_results.csv${NC} & ${BLUE}aggregated_summary.csv${NC}"
echo -e "  - Figures: ${BLUE}results/figures/*.png${NC} (15 publication-grade figures)"
echo -e "  - Models:  ${BLUE}models/${NC} (Trained forecaster artifacts)"
echo -e "==============================================================================\n"
