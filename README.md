# Predictive Maintenance — End-to-End ML System

An end-to-end predictive maintenance system built with:

- FastAPI (inference service + model lifecycle API)
- Streamlit (interactive dashboard + monitoring)
- Postgres + pgvector (storage, registry, vector search ready)
- MLflow (experiment tracking)
- Docker Compose (container orchestration)

This project demonstrates:
- Time-series feature engineering (rolling window statistics)
- Remaining Useful Life (RUL) regression modeling
- ML experiment tracking (MLflow)
- Model registry with staged promotion (dev → prod)
- Secure model promotion via API
- Live model reload without container restart
- Prediction logging
- Feature drift detection (PSI)
- Containerized deployment
- (Planned) Retrieval-Augmented Generation (RAG) assistant

---

## Architecture

Streamlit (UI) → FastAPI (Model Inference API) → Postgres  
Training Pipeline → MLflow → Model Registry → Production Model

All services run via Docker Compose.

Python 3.12.x required
Docker images use `python:3.12-slim`

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
```powershell
Invoke-RestMethod http://localhost:8000/model/info
```

#### The API always serves from:
`/artifacts/models/production.pkl`

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

Prediction logs are stored in Postges and analyzed in real time.
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
- [x] Model registry with staged promotion
- [x] Monitoring + drift detection (PSI)
- [ ] Structured model benchmarking framework
- [ ] PyTorch deep learning sequence model (LSTM/TCN)
- [ ] Monitoring & drift detection
- [ ] RAG assistant with pgvector
- [ ] Cloud deployment (AWS/Azure)

---

## Author

Mark Aikman



