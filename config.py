"""
config.py  –  Central configuration for the Predictive Auto-Scaling project.
Every tunable constant lives here so nothing is hard-coded elsewhere.
"""
from pathlib import Path

# ── Project root (wherever this file sits) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Directory layout ───────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "models"
RESULTS_DIR        = PROJECT_ROOT / "results"
PLOTS_DIR          = RESULTS_DIR  / "plots"
PREDICTIONS_DIR    = RESULTS_DIR  / "predictions"
SIMULATION_DIR     = RESULTS_DIR  / "simulation"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
           RESULTS_DIR, PLOTS_DIR, PREDICTIONS_DIR, SIMULATION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset generation ─────────────────────────────────────────
# Set LITE_MODE = True for a quick test run (~50 MB per file).
# Set LITE_MODE = False for the full 1 GB+ run (takes more time & disk).
LITE_MODE = True

DAYS              = 30           # simulation horizon
INTERVAL_MINUTES  = 5            # time-step resolution
INTERVALS_PER_DAY = (24 * 60) // INTERVAL_MINUTES  # 288

# machines × days × 288 = total rows per raw dataset
DATASETS = {
    "d1_alibaba_2018":     {"machines": 200  if LITE_MODE else 2000, "has_gpu": False},
    "d2_azure_2019":       {"machines": 200  if LITE_MODE else 2000, "has_gpu": False},
    "d3_alibaba_gpu_2020": {"machines": 100  if LITE_MODE else 1000, "has_gpu": True},
}

# ── Train / Validation / Test split ───────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# ── Forecasting hyper-parameters ──────────────────────────────
MA_WINDOWS          = [6, 12, 24, 48]       # candidates for SMA window
LR_LAGS             = 24                     # lag features for Linear Reg.
HW_SEASONAL_PERIODS = INTERVALS_PER_DAY     # 288 (one full day)
GB_LAGS             = 24
GB_ROLLING_WINDOWS  = [6, 12, 24, 48]
GB_N_ESTIMATORS     = 200
GB_MAX_DEPTH        = 5
GB_LEARNING_RATE    = 0.1

# ── Cluster-simulator parameters ─────────────────────────────
INITIAL_NODES          = 10
MIN_NODES              = 2
MAX_NODES              = 50
NODE_CAPACITY          = 100.0     # CPU-units per node (updated by Spark benchmark)
SCALE_UP_THRESHOLD     = 0.75      # reactive: scale out when util > 75 %
SCALE_DOWN_THRESHOLD   = 0.35      # reactive: scale in  when util < 35 %
PRED_UP_THRESHOLD      = 0.65      # predictive: scale out earlier at 65 %
PRED_DOWN_THRESHOLD    = 0.40      # predictive: scale in  later  at 40 %
COOLDOWN_STEPS         = 3         # 3 steps = 15 min between actions
COST_PER_NODE_PER_STEP = 0.01      # $/node/5-min  (≈ $2.88/node/day)
SLA_LATENCY_LIMIT      = 1.0       # normalised; >1 = SLA breach

# ── Reproducibility ──────────────────────────────────────────
RANDOM_SEED = 42
