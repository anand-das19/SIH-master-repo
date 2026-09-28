# Smart Irrigation Predictive Model — Final Strategy & Execution Plan

## 1. Executive Summary & Core Decisions

### Dataset Decision:
- We will not blindly merge all dataset files together.
- We will use dataset/data.csv (16,411 rows, clean, zero missing values) as the sole training dataset for our MVP.

### Feature Scope (Laser-Focused MVP):
- **Inputs (3 Features)**:
  - MOI: Soil Moisture %
  - temp: Ambient Temperature (°C)
  - humidity: Relative Humidity (%)
- **Target (Binary Classification)**:
  - `need_irrigation`:
    - `0`: No Irrigation (Pump OFF — 0 mins)
    - `1`: Irrigation Required (Pump ON — mapped to duration by deficit)
  - Natural binary distribution: ~55% Class 0 (9,062 rows) vs. ~45% Class 1 (7,349 rows).

### Dual-Model Strategy (ML + DL):
- **Model 1**: Gradient Boosted Trees (XGBoost / LightGBM) — Lightweight, highly accurate, near-zero latency, production edge ready.
- **Model 2**: Deep Neural Network (ANN / Multi-Layer Perceptron) — Deep Learning baseline to capture non-linear interactions and provide a head-to-head comparison benchmark for judges.

---

## 2. End-to-End Architecture

```
                 [ SENSOR INPUTS ]
      Soil Moisture (MOI) | Temperature | Humidity
                          │
                          ▼
            ┌───────────────────────────┐
            │   Data Cleaning & Splits  │
            │   (StandardScaler / MinMax│
            │   Stratified Train / Test)│
            └─────────────┬─────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │ Model 1: XGBoost /   │    │ Model 2: Deep Neural │
 │ LightGBM Classifier  │    │ Network (ANN / MLP)  │
 └──────────┬───────────┘    └──────────┬───────────┘
            │                           │
            └─────────────┬─────────────┘
                          │ (Model Comparison & Export)
                          ▼
            ┌───────────────────────────┐
            │   Decision & Action Engine│
            │  Class 0: Pump OFF (0 min)│
            │  Class 1: Pump ON (15-40m)│
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │      demo_predict.py      │
            │ (Interactive CLI Demo     │
            │  for Judges & Teammates)  │
            └───────────────────────────┘
```

---

## 3. Step-by-Step Phase Execution Plan

### Phase 1: Ingestion & Exploratory Data Analysis (EDA)
- Load and profile dataset/data.csv (16,411 rows).
- Verify statistical distributions of MOI, temp, and humidity across classes 0 and 1 (Binary).
- Check correlation metrics and identify critical moisture cutoff thresholds.
- Save a clean split dataset artifact.

### Phase 2: Preprocessing & Formulation
- Apply feature scaling (StandardScaler / MinMaxScaler).
- Create a stratified train/validation/test split (70% / 15% / 15%) to preserve class balance.
- Set up evaluation pipelines for binary classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

### Phase 3: Dual-Model Training & Benchmarking
- Train Model 1 (XGBoost / LightGBM) with hyperparameter tuning.
- Train Model 2 (Deep Neural Network / MLP) with Dropout and BatchNorm.
- Generate a Benchmark Comparison Table (Accuracy, Precision, Recall, F1, ROC-AUC, Latency) to use directly in presentation slides.
- Export serialized production artifacts: smart_irrigation_xgb.joblib and smart_irrigation_ann.pt (or .pkl).

### Phase 4: Decision Engine & Demonstration Suite
- Build rule-to-runtime mapper:
  - Class 0 → Pump OFF (0 mins)
  - Class 1 → Pump ON (runtime dynamically scaled by moisture deficit: 15-40 mins)
- Build demo_predict.py: A clean command-line interface where anyone (judges, teammates, or frontend) can input:
  python demo_predict.py --moi 32 --temp 35 --humidity 45  And get an immediate, formatted verdict:
  [Result] Decision: IRRIGATION REQUIRED (Class 1) | [Action] Run pump for 40 minutes. | [Latency] 1.4 ms | Confidence: 97.2%
---

## 4. Why This Plan Wins the Hackathon
1. **Ultra-Fast Development**: Takes minimal time to build and verify, leaving ample bandwidth to build remaining models.
2. **Judge Appeal**: Demonstrates both Classical ML (XGBoost) and Deep Learning (ANN) side-by-side with hard benchmark numbers.
3. **Demo-Ready**: Delivers an interactive demo_predict.py that proves real-world utility immediately without breaking on uncalibrated hardware data.
