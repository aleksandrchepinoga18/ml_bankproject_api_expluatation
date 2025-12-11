# ML Bank Scoring Pipeline

Production-ready ML system for bank risk scoring with:
- Flask API for real-time predictions
- Drift monitoring (PSI, KS-test)
- Automatic retraining on data drift
- Quality tracking (ROC-AUC, F1)

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API
```bash
python app/api.py
```

### Test API (example)
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"first_tx_timestamp": 1615161978.0, "last_tx_timestamp": 1627349954.0, ...}'
```

## 🔍 Monitoring

### Check for drift and retrain if needed
```bash
python -m monitoring.retrain_if_needed
```

### Simulate labels (for testing only)
```bash
python monitoring/simulate_labels.py
```

## 📁 Project Structure
- `app/` — Flask API
- `src/` — Model training pipeline
- `monitoring/` — Drift detection, quality, retraining
- `models/` — Saved models (not in Git)
- `data/` — Raw data (not in Git)

## ⚠️ Note
- Data and models are excluded from Git (see `.gitignore`)
- Use environment variables for database credentials in production
