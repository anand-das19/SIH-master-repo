# Smart Irrigation Model

## Overview

This repository implements a **smart irrigation system** that predicts whether irrigation is required based on three sensor inputs:
- **Moisture (MOI)** – Soil moisture percentage
- **Temperature** – Ambient temperature in °C
- **Humidity** – Relative humidity percentage

The pipeline is split into three phases:
1. **Ingestion & EDA** – Load raw sensor data, perform exploratory data analysis and generate an HTML report.
2. **Pre‑processing** – Stratified train/validation/test split, feature scaling, and persisting the scaler.
3. **Model Training & Inference** – Train two models (a shallow `XGBoost` classifier and a deeper PyTorch ANN) and provide a CLI demo for runtime prediction.

All code is written in **Python 3.11** and relies on `pandas`, `numpy`, `scikit‑learn`, `joblib`, `torch`, and `xgboost`.

---

## Repository Structure

```
smart irrigation model/
├─ architechture.txt                 # High‑level system diagram
├─ dataset/                          # Data files (raw, cleaned, splits)
│   ├─ cleaned_binary_irrigation.csv
│   ├─ train.csv
│   ├─ val.csv
│   ├─ test.csv
│   └─ processed_splits.joblib      # Scaled feature bundles
├─ models/                           # Trained model artifacts
│   ├─ scaler.joblib                  # StandardScaler fitted on training data
│   ├─ smart_irrigation_ann.pt        # PyTorch ANN checkpoint
│   └─ smart_irrigation_xgb.joblib    # XGBoost classifier
├─ reports/                          # Generated reports (EDA HTML, metrics)
├─ src/                              # Source code
│   ├─ __pycache__/                  # Compiled byte‑code (ignored)
│   ├─ demo_predict.py               # CLI demo for inference
│   ├─ eda_ingestion.py              # Phase 1 – ingestion & EDA
│   ├─ evaluate_metrics.py           # Phase 3 – compute model metrics
│   ├─ generate_eda_report.py        # Helper to create HTML EDA report
│   ├─ preprocess.py                 # Phase 2 – preprocessing & scaler
│   └─ train_models.py               # Phase 3 – training both models
├─ tests/                            # Unit / integration tests
├─ detail_phase_wise_plan.md         # Original design plan
└─ README.md                         # **This file**
```

---

## Core Functionality

### `src/eda_ingestion.py`
- Loads the raw CSV (`dataset/cleaned_binary_irrigation.csv`).
- Performs basic cleaning, type conversion, and optional outlier removal.
- Calls `generate_eda_report.py` to produce an interactive HTML report (`reports/eda_report.html`).

### `src/preprocess.py`
- Executes **Phase 2** preprocessing (`run_phase2_preprocessing`).
- Performs a **stratified** 70/30 train‑temp split, then splits the temporary set evenly into validation and test sets (15 % each).
- Fits a `StandardScaler` **only on the training split** (leak‑free) and saves it to `models/scaler.joblib`.
- Persists raw CSV splits for transparency and a bundled NumPy payload (`processed_splits.joblib`) for fast downstream ingestion.

### `src/train_models.py`
- Loads the processed bundles.
- Trains two independent models:
  1. **XGBoost** (`smart_irrigation_xgb.joblib`) – a gradient‑boosted decision tree classifier.
  2. **ANN** (`smart_irrigation_ann.pt`) – a three‑layer fully‑connected neural network built with PyTorch.
- Saves both model artifacts under `models/`.

### `src/evaluate_metrics.py`
- Computes common classification metrics (accuracy, precision, recall, F1) on the validation and test sets.
- Optionally produces ROC curves saved in `reports/`.

### `src/demo_predict.py`
- Provides a **CLI demo** (`python -m src.demo_predict`) that:
  1. Loads the scaler and the requested model (ANN or XGBoost).
  2. Scales the supplied sensor values.
  3. Predicts the irrigation class and probability.
  4. Maps a **Class 1** prediction to a runtime (15‑40 min) based on the moisture level using a linear inverse mapping.
  5. Prints a nicely formatted result with latency statistics.

---

## Getting Started

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt   # (or manually: pandas, numpy, scikit-learn, joblib, torch, xgboost)

# 3. Run Phase 1 – ingestion & EDA
python -m src.eda_ingestion

# 4. Run Phase 2 – preprocessing
python -m src.preprocess

# 5. Train both models (Phase 3)
python -m src.train_models

# 6. Evaluate metrics (optional)
python -m src.evaluate_metrics

# 7. Demo inference (choose model)
python -m src.demo_predict --moi 30 --temp 28 --humidity 55 --model ann
```

---

## Future Enhancements (Roadmap)

> [!NOTE] The following items are **planned** but not yet implemented. They can be prioritized based on project goals.

1. **Automated Hyper‑parameter Search** – Integrate Optuna / Ray Tune to automatically find the best model parameters for both XGBoost and the ANN.
2. **Model Registry & Versioning** – Store models in a lightweight MLflow‑compatible registry to track experiments and enable roll‑backs.
3. **Edge Deployment Scripts** – Generate a TinyML‑compatible version of the ANN (e.g., TensorFlow Lite) for deployment on micro‑controllers.
4. **Streaming Data Ingestion** – Replace static CSV ingestion with a real‑time MQTT consumer that logs sensor data directly to the preprocessing pipeline.
5. **Dashboard UI** – Build a small Streamlit or FastAPI dashboard to visualise live predictions, historical irrigation decisions, and model performance over time.
6. **Explainability** – Add SHAP / LIME visualisations to explain individual predictions, especially for the XGBoost model.
7. **Continuous Integration** – Add GitHub Actions for linting, unit tests, and automatic model artifact publishing.
8. **Dockerisation** – Provide a `Dockerfile` and `docker‑compose.yml` to run the entire pipeline in containers, simplifying reproducibility.
9. **Multi‑sensor Fusion** – Incorporate additional sensors (e.g., soil temperature, solar irradiance) and explore feature engineering pipelines.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. Follow the standard **PEP 8** style guide and ensure that any new code is covered by tests under the `tests/` directory.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.
