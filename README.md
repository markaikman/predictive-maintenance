# Predictive Maintenance — End-to-End ML System

An end-to-end predictive maintenance system built with:

- **FastAPI** (inference service + model lifecycle API)
- **Streamlit** (interactive dashboard + monitoring)
- **Postgres** + **pgvector** (storage, registry, vector search ready)
- **MLflow** (experiment tracking)
- **Docker Compose** (container orchestration)

This project demonstrates:
- Time-series feature engineering (rolling window statistics)
- Remaining Useful Life (RUL) regression modeling
- Structured model benchmarking across multiple splits and feature windows
- ML experiment tracking (MLflow)
- Model registry with staged promotion (`dev` → `prod`)
- Secure model promotion via API
- Live model reload without container restart
- Prediction logging
- Feature drift detection (PSI)
- Containerized deployment
- (Planned) Retrieval-Augmented Generation (RAG) assistant

---

## Architecture
```SCSS
Streamlit (UI)
      ↓
FastAPI (Inference + Model Lifecycle API)
      ↓
Postgres (Prediction Logs + Model Registry)
      ↑
Training Pipeline → MLflow → Model Artifacts
```

The API always serves the active production model from:

```bash
/artifacts/models/production.pkl
```

All services run via Docker Compose.
- Python 3.12.x required
- Docker images use `python:3.12-slim`

---

## Model Lifecycle
This system supports staged model deployment:
- `dev` – latest trained model
- `prod` – actively served production model  
Training automatically registers a model as active `dev`

### Promote dev → prod
```powershell
$headers = @{ "X-API-KEY" = "your_admin_key" }
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/model/promote" -Headers $headers
```

### Check current model
`/model/info` reports:
- `loaded_run_id`
- `registry_prod_run_id`
- `loaded_matches_registry_prod`  
This confirms the API is serving the active production run.
```powershell
Invoke-RestMethod http://localhost:8000/model/info
```

#### The API always serves from:
`/artifacts/models/production.pkl`

---
## Structured Model Benchmarking
The benchmarking pipeline evaluates models across:  
- Multiple rolling windows (e.g., 5, 10, 20, 30)
- Multiple engine-level train/validation splits (different seeds)
- Multiple model families:
    - Ridge
    - Random Forest
    - Gradient Boosting
    - HistGradientBoosting
    - XGBoost
    - LightGBM

Models are ranked by: 
- Mean RMSE across splits (primary)
- Mean MAE (tie-breaker)

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:DATABASE_URL="postgresql+psycopg2://dsuser:password@localhost:5432/dsdb"

python -m pipelines.benchmark.benchmark_fd001
```
Outputs:
- Per-run leaderboard CSV
- Aggregated leaderboard (mean/std metrics)
- Summary JSON
- MLflow runs for every model configuration
- Automatic registration of best configuration as active `dev`

---
## Monitoring
The Streamlit dashboard includes a Monitoring tab showing:
- Prediction volume over time
- Prediction distribution
- Feature drift vs training baseline using Population Stability Index (PSI)

PSI Interpretation:
- < 0.10 → Low drift
- 0.10–0.25 → Moderate drift
- 0.25 → High drift

Prediction logs are stored in Postgres and analyzed in real time.  
Baseline stats are versioned by `run_id` and aligned at promotion.
___

## Data

This project uses the [NASA C-MAPSS Jet Engine Simulated Degradation Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) (FD001).

Note: If the official NASA listing is unavailable, use any public mirror of: 
- `train_FD001.txt` 
- `test_FD001.txt`
- `RUL_FD001.txt`

Place files in `data/raw/`.

---

## Training
Train the baseline RUL model:
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:DATABASE_URL="postgresql+psycopg2://dsuser:password@localhost:5432/dsdb"

python pipelines/train/train_rul_fd001.py
```

Training will:
- Log metrics to MLflow
- Save model bundle
- Save basline drift statistics
- Register the model as active `dev`

---

## Model Benchmarking & Tuning
This repo includes scripts to compare models and tune LightGBM with robust multi-seed evaluation.

### Validate current best LightGBM params (multi-seed)
Runs the tuned parameter set across multiple engine-based train/val splits to measure stability.

```powershell
python pipelines/validate/validate_best_lgbm_fd001.py
```

Outputs a CSV to `artifacts/` and prints mean/std across seeds.

### Multi-seed LightGBM tuning
Searches hyparparameters and selects teh best trial by:
1. lowest `RMSE_mean`
2. lowest `RMSE_std` (tie-break)

```powershell
$env:N_TRIALS="40"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:DATABASE_URL="postgresql+psycopg2://maikman:password@localhost:5432/dsdb"
python -m pipelines.tune.tune_lgbm_multiseed_fd001
```

The best run is registered as active `dev`. Promote via the API when ready:  
`POST /model/promote`

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Create environment file

Copy the example file:

```bash
cp .env.example .env
```

Ensure `.env` contins:
```ini
DATABASE_URL=...
ADMIN_API_KEY=...
```

### 3. Start services

```bash
docker compose up --build
```

### 4. Access services

- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

---

## Development Setup (Optional Local Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

## Roadmap

- [x] Data ingestion + feature engineering
- [x] Baseline predictive model (RUL)
- [x] Structured benchmarking framework (multi-window, multi-split)
- [x] Model registry with staged promotion
- [x] Monitoring + drift detection (PSI)
- [x] Hyperparameter tuning with multi-seed evaluation (LightGBM)
- [ ] PyTorch deep learning sequence model (LSTM/TCN)
- [ ] RAG assistant with pgvector
- [ ] Cloud deployment (AWS/Azure)

---

## Author

Mark Aikman



