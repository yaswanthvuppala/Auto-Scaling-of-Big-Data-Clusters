# Predictive Auto-Scaling of Big-Data Clusters using Workload Forecasting

A complete cloud computing research pipeline evaluating **Predictive vs. Reactive Auto-Scaling** on distributed Big-Data clusters (Apache Spark / Hadoop YARN / Kubernetes) across three large-scale traces:
* **D1:** Alibaba Cluster Trace 2018 (`container_usage` / `batch_instance`)
* **D2:** Azure Public Dataset V2 2019 (`vm_cpu_readings`)
* **D3:** Alibaba GPU Cluster Trace 2020 (`task_usage`)

---

## 1. Problem Statement & Motivation

Commercial cloud auto-scalers (e.g., Kubernetes Horizontal Pod Autoscaler, AWS Auto Scaling) use **reactive rules**:
$$\rho_t = \frac{\text{Demand}_t}{\text{Capacity}_t} > \tau_{\text{up}} \implies \text{Scale Out}$$
Because virtual machine and container provisioning incurs a physical spin-up latency (typically 2–5 minutes for VM boot, runtime configuration, and Spark worker registration), reactive scaling reacts **after** the load has already risen. During this initialization window, the cluster is under-provisioned, queue backlog surges, and **SLA violations** occur.

**Proposed Solution:** Deploy lightweight time-series forecasters (Moving Average, Linear Regression, Holt-Winters, Gradient Boosted Trees) that forecast cluster demand $h$ steps ahead ($h \ge \text{provisioning delay}$), proactively initiating node scale-out **before** workload peaks arrive.

---

## 2. Project Directory Structure

```text
Cloud-Project/
├── setup_project.sh           # Unix/macOS directory initialization script
├── setup_project.ps1          # Windows PowerShell directory initialization script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes raw data, bytecode, and temporary logs
├── README.md                  # Project documentation & execution guide
├── report.md                  # Complete 5–7 page academic research report
│
├── data/
│   ├── raw/
│   │   ├── d1_alibaba2018/    # Alibaba 2018 container_usage traces (>= 1 GB)
│   │   ├── d2_azure2019/      # Azure 2019 VM CPU readings (>= 1 GB)
│   │   └── d3_alibaba_gpu2020/# Alibaba GPU 2020 task_usage traces (>= 1 GB)
│   ├── parquet/               # Optimized snappy-compressed columnar Parquet files
│   └── processed/             # Clean aggregated 5-min workload series (D1.csv, D2.csv, D3.csv)
│
├── src/
│   ├── download.py            # Official trace downloader (curl/wget resume) & synthetic generator
│   ├── inspect_data.py        # PySpark explicit schema verification (no inferSchema)
│   ├── preprocess_spark.py    # Large-scale PySpark ETL & 5-min aggregation pipeline
│   ├── split.py               # 60/20/20 chronological split & zero-leakage unit test
│   ├── benchmark_spark.py     # Physical PySpark throughput benchmark across cores
│   ├── simulator.py           # Time-step cluster simulator with M/M/1 queuing latency
│   ├── predictors.py          # Unified forecasting suite (Naive, MA, LR, HW, LightGBM/HistGB)
│   ├── scalers.py             # Elasticity policies (Reactive, Predictive, Hybrid)
│   ├── metrics.py             # SLA violation rate, P95 latency, cost, and mean +/- std
│   ├── run_experiments.py     # Full grid runner: policies x datasets x delays x margins
│   └── plots.py               # 300 DPI publication figures (EDA & experimental graphs)
│
├── results/
│   ├── figures/               # High-resolution plots (PNG @ 300 DPI)
│   └── tables/                # Benchmark tables, forecasting errors, and all_results.csv
```

---

## 3. Quick Start & Execution Guide

### Step 1: Environment Setup
```powershell
# Windows PowerShell
.\setup_project.ps1
pip install -r requirements.txt
```
```bash
# Linux / macOS
chmod +x setup_project.sh && ./setup_project.sh
pip install -r requirements.txt
```

### Step 2: Download or Generate Raw Data (>= 1 GB per dataset)
```bash
# Option A: Synthesize contiguous >= 1 GB traces with exact official schemas:
python src/download.py --mode realistic

# Option B: Print direct official download links with resume support:
python src/download.py --mode info
```

### Step 3: Inspect Raw Schemas (PySpark)
```bash
python src/inspect_data.py
```

### Step 4: Run Preprocessing & Parquet Conversion
```bash
python src/preprocess_spark.py
```

### Step 5: Generate Exploratory Data Analysis (EDA) Figures
```bash
python -c "from src.plots import generate_eda_plots; generate_eda_plots()"
```

### Step 6: Measure Spark Core Throughput & Calibrate Node Capacity
```bash
python src/benchmark_spark.py
```

### Step 7: Execute Full Experimental Grid (Simulations + Predictions + Final Plots)
```bash
python src/run_experiments.py
```

---

## 4. Workload Forecasting Models

| Predictor | Family | Feature Engineering / Formulation | Tuning on Val |
| :--- | :--- | :--- | :--- |
| **Naive** | Baseline | Last known observation: $\hat{y}_{t+h} = y_t$ | None |
| **Moving Average** | Statistical | Rolling mean of last $k$ observations | $k \in \{3, 6, 12, 24, 48\}$ |
| **Linear Regression** | Supervised | 24 autoregressive lag features | Lag window length |
| **Holt-Winters** | Classical TS | Level + Trend + Seasonality ($m = 288$ steps = 24 hrs) | Smoothing constants $\alpha, \beta, \gamma$ |
| **Gradient Boosting** | Ensemble ML | Lags + Rolling stats + Cyclical $[\sin, \cos]$ calendar encodings | Trees, depth, learning rate |

---

## 5. Elasticity Policies

1. **Reactive Scaler:**
   - Evaluates current load at step $t$.
   - Scale out if utilization $\rho_t > 80\%$.
   - Scale in if utilization $\rho_t < 40\%$ (enforces 3-step cooldown to prevent thrashing).
2. **Predictive Scaler:**
   - Evaluates forecasted demand for step $t+h$ ($h \ge \text{provisioning delay}$).
   - Provisions nodes proactively:
     $$\text{Nodes} = \left\lceil \frac{\hat{D}_{t+h}}{\text{NodeCapacity} \times \tau_{\text{target}}} \times (1 + \text{safety\_margin}) \right\rceil$$
3. **Hybrid Scaler:**
   - Predictive scale-out as primary controller.
   - Reactive safety net triggers emergency provisioning if current load surges unexpectedly beyond $85\%$.

---

## 6. M/M/1 Queuing Latency Formulation

In cloud clusters, processing latency escalates asymptotically under congestion:
$$\rho = \min\left(\frac{\text{Demand}}{\text{Capacity}}, 0.98\right), \quad \text{Latency} = \frac{1 - \tau_{\text{target}}}{1 - \rho} \quad (\tau_{\text{target}} = 0.75)$$
* $\rho \le 75\% \implies \text{Latency} \le 1.0$ (Compliant with SLA).
* $\rho > 75\% \implies \text{Latency} > 1.0$ (SLA Violation).
* Any unserviced demand enters a **backlog queue**, directly inflating latency in subsequent steps.

---

## 7. License & Citation
Developed for academic cloud computing research and individual viva voce defense. References Alibaba Cluster Data 2018/2020 and Microsoft Azure Public Dataset 2019 repositories.
