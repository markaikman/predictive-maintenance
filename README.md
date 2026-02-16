# Predictive Maintenance — End-to-End ML System

An end-to-end predictive maintenance system built with:

- FastAPI (inference service)
- Streamlit (interactive dashboard)
- Postgres + pgvector (storage + vector search)
- MLflow (experiment tracking)
- Docker Compose (container orchestration)

This project demonstrates:
- Feature engineering from time-series data
- Predictive modeling (regression/classification)
- Model tracking and versioning
- Containerized deployment
- Prediction logging and monitoring foundation
- (Planned) Retrieval-Augmented Generation (RAG) assistant

---

## Architecture

Streamlit (UI) → FastAPI (Model Inference API) → Postgres  
Training Pipeline → MLflow → Model Artifacts → API loads latest model  

All services run via Docker Compose.

Python 3.12.x required
Docker images use python:3.12-slim

---

## Data

This project uses the NASA C-MAPSS turbofan engine degradation dataset (FD001).

Note: the data.nasa.gov listing may be unavailable at times. If so, use an alternative public mirror of the same C-MAPSS files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) and place them in `data/raw/`.

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

Edit values if needed.

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

- [ ] Data ingestion + feature engineering
- [ ] Baseline predictive model
- [ ] PyTorch deep learning model
- [ ] Monitoring & drift detection
- [ ] RAG assistant with pgvector
- [ ] Cloud deployment (AWS/Azure)

---

## Author

Mark Aikman



