# 🌾 AgriNode AI — Environmental Risk Intelligence Model

> **Latitude + Longitude + Date → Calibrated Multi-Hazard Risk Engines → Actionable Farm Decisions**

An explainable, empirically benchmarked environmental risk intelligence system built for Indian agriculture. The model ingests historical climate data, resolves any geographic coordinate to its nearest district, computes calibrated drought, flood, and heat stress risk scores, synthesizes them into a multi-hazard assessment, and outputs **concrete, operational farm decisions** — not just abstract numbers.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [System Architecture](#system-architecture)
- [Risk Engines](#risk-engines)
  - [Drought Risk Engine](#1-drought-risk-engine)
  - [Flood Risk Proxy Engine](#2-flood-risk-proxy-engine)
  - [Heat Stress Engine](#3-heat-stress-engine)
  - [Multi-Hazard Synthesis](#4-multi-hazard-synthesis)
- [Farm Decision Engine](#farm-decision-engine)
- [Feature Engineering](#feature-engineering)
- [Geographic Resolution](#geographic-resolution)
- [Datasets](#datasets)
- [Benchmarking & Validation](#benchmarking--validation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Future Scope & Enhancements](#future-scope--enhancements)
- [License](#license)

---

## Key Highlights

- **Three Calibrated Risk Engines** — Drought, Flood Proxy, and Heat Stress, each producing continuous scores (0–100) with transparent explainability factors.
- **Multi-Hazard Synthesis** — Severity-dominance aggregation logic that mirrors how real disaster management frameworks operate.
- **Farm Decision Engine** — Translates risk tiers into concrete irrigation orders, field operation directives, and time-sensitive advisories.
- **Sub-Millisecond Geo Resolution** — KDTree-based coordinate-to-district mapping covering 700+ Indian districts, zero GDAL/GIS dependencies.
- **Empirical Ground-Truth Benchmarking** — Validated against 21 curated historical Indian extreme weather events (2015–2022) with **≥85% disaster recall** and **≤15% false alarm rate**.
- **Explainable AI** — Every risk assessment comes with human-readable "Why?" factors explaining the score drivers.
- **Lightweight & Portable** — Pure Python, no web servers, no complex GIS libraries, runs on any OS.

---

## System Architecture

```text
                                  INPUT
                       Latitude, Longitude, Date
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Geographic Resolver    │
                       │   (src/geo_resolver.py)  │
                       │   KDTree Centroid Match   │
                       └────────────┬────────────┘
                                    │ District, State
                                    ▼
                       ┌─────────────────────────┐
                       │  Historical Baseline &   │
                       │   Feature Engineering    │
                       │    (src/features.py)     │
                       │ Rolling Windows & Z-Norm │
                       └────────────┬────────────┘
                                    │ Feature Vector
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
  │   DROUGHT    │           │ FLOOD PROXY  │           │ HEAT STRESS  │
  │    ENGINE    │           │    ENGINE     │           │    ENGINE    │
  │(src/drought) │           │ (src/flood)   │           │  (src/heat)  │
  └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │   Multi-Hazard Engine    │
                       │  (src/overall_risk.py)   │
                       │ Calibrated Risk Levels   │
                       └────────────┬────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │  Farm Decision Engine    │
                       │(src/decision_engine.py)  │
                       │ Actionable Field Orders  │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐                   ┌─────────────────────────────────┐
│        src/benchmark.py         │                   │       src/demo_predict.py       │
│  Empirical Testing Suite        │                   │   Interactive CLI Demo           │
│  • 21 Disaster Benchmarks       │                   │   • Preset Case Studies          │
│  • Recall, Specificity, Latency │                   │   • Interactive Lat/Lon Input    │
│  • Quantitative Report          │                   │   • Actionable Advisory Output   │
└─────────────────────────────────┘                   └─────────────────────────────────┘
```

---

## Risk Engines

All risk engines output a **continuous score (0–100)** and a **discrete tier**:

| Score Range | Tier         |
|-------------|--------------|
| 0 – 30      | **LOW**      |
| 31 – 60     | **MODERATE** |
| 61 – 80     | **HIGH**     |
| 81 – 100    | **SEVERE**   |

### 1. Drought Risk Engine

**File:** `src/drought.py`

Evaluates multi-scale rainfall deficit severity using a weighted composite:

| Component | Weight | Description |
|-----------|--------|-------------|
| 90-day rainfall anomaly | 40% | Long-term cumulative precipitation deficit relative to historical normal |
| 30-day rainfall anomaly | 30% | Medium-term monsoon performance tracking |
| Dry-spell persistence | 30% | Consecutive days with rainfall < 2.5 mm, scaled by severity |

### 2. Flood Risk Proxy Engine

**File:** `src/flood.py`

Estimates acute surface water excess and soil saturation risk:

| Component | Weight | Description |
|-----------|--------|-------------|
| 1-day peak rainfall vs percentiles | 50% | Compares daily rainfall against historical 95th and 99th percentiles |
| 7-day soil saturation accumulation | 30% | Weekly cumulative precipitation indicating ground saturation |
| Heavy-rain day frequency | 20% | Count of extreme rainfall days in the past 7 days |

> **Note:** This is explicitly a *flood risk proxy* — it captures excess precipitation and surface runoff signals. Actual flood modeling requires hydrological and terrain data.

### 3. Heat Stress Engine

**File:** `src/heat.py`

Captures thermal anomaly magnitude and heatwave persistence:

| Component | Weight | Description |
|-----------|--------|-------------|
| Local baseline departure | 40% | °C deviation above the district's historical monthly mean Tmax |
| Absolute thermal threshold | 30% | Scoring against fixed physiological danger thresholds (40°C, 45°C, 50°C) |
| Heatwave persistence | 30% | Consecutive days exceeding the local 90th percentile Tmax |

### 4. Multi-Hazard Synthesis

**File:** `src/overall_risk.py`

Aggregates individual hazard tiers using **severity dominance logic**:

```
If ANY hazard is SEVERE     → Overall = SEVERE
Elif ANY hazard is HIGH     → Overall = HIGH
Elif ANY hazard is MODERATE → Overall = MODERATE
Else                        → Overall = LOW
```

The **primary hazard** is identified as the individual engine with the highest raw score.

---

## Farm Decision Engine

**File:** `src/decision_engine.py`

The model doesn't stop at risk scores — it translates them into **concrete, operational field commands**:

| Risk Scenario | Irrigation Orders | Field Operations |
|---------------|-------------------|------------------|
| **Drought HIGH/SEVERE** | Trigger deficit drip irrigation targeting root-zone replenishment | Apply organic mulch; suspend non-essential fertilizer; consider emergency foliar spray |
| **Flood HIGH/SEVERE** | Immediate shutdown of all pumps and irrigation lines | Open drainage channels; clear furrows; prepare anti-fungal spray for post-recession |
| **Heat HIGH/SEVERE** | Schedule early-morning light canopy cooling cycle | Shift labor before 10:00 AM; stop mid-day chemical spraying; deploy shade nets |
| **Low/Moderate Risk** | Normal irrigation schedule | Routine seasonal maintenance and monitoring |

The engine outputs a **unified action plan** with priority ranking (CRITICAL → HIGH → ROUTINE), automatically de-duplicating routine actions when elevated alerts exist.

---

## Feature Engineering

**File:** `src/features.py`

Risk is calculated strictly **relative to local historical norms**, not absolute thresholds. The feature engineering pipeline computes:

### Rainfall Features
| Feature | Description |
|---------|-------------|
| `rain_1d` | Latest single-day rainfall (mm) |
| `rain_7d` | 7-day cumulative precipitation |
| `rain_30d` | 30-day cumulative precipitation |
| `rain_90d` | 90-day cumulative precipitation |
| `anomaly_30d` | % departure from 30-day historical baseline |
| `anomaly_90d` | % departure from 90-day historical baseline |
| `current_dry_spell` | Consecutive days with rainfall < 2.5 mm |
| `heavy_rain_days_7d` | Days exceeding 95th percentile in last 7 days |

### Temperature Features
| Feature | Description |
|---------|-------------|
| `tmax` | Daily maximum temperature (°C) |
| `tmin` | Daily minimum temperature (°C) |
| `tmax_anomaly` | °C departure from historical monthly mean Tmax |
| `consecutive_hot_days` | Days exceeding local 90th percentile threshold |
| `max_tmax_7d` | Maximum Tmax observed in the past 7 days |

### Historical Baselines
Pre-computed per district per calendar month from historical records (1981–2015):
- Monthly mean and standard deviation of rainfall and Tmax
- 90th, 95th, and 99th percentile thresholds for rainfall
- 90th percentile threshold for Tmax (heatwave detection)

Stored in `data/processed/district_baselines.csv`.

---

## Geographic Resolution

**File:** `src/geo_resolver.py`

Resolves any (latitude, longitude) pair to the nearest Indian district in **< 1 ms**:

- **700+ district centroids** pre-compiled in `data/processed/district_centroids.csv`
- Uses `scipy.spatial.KDTree` with 3D spherical (Cartesian) coordinates for accurate nearest-neighbor search
- Computes exact Haversine great-circle distance to the matched centroid
- **Zero GDAL/GeoPandas dependencies** — works on any OS without complex GIS library installation

---

## Datasets

| Dataset | Source | Role |
|---------|--------|------|
| **IMD Daily District Rainfall** | India Meteorological Department (NWDP Archive) | Primary precipitation source for drought and flood risk |
| **INDmet Daily District Temperature** | Zenodo 1981–2024 (~153 MB) | Tmax/Tmin records for heat stress modeling |
| **District Centroids** | Pre-compiled coordinate registry | Geographic resolution lookup table |
| **District Baselines** | Computed from historical records | Monthly climate normals and percentile thresholds |
| **Benchmark Cases** | 21 curated historical events | Ground-truth validation dataset |

All raw data is stored in `data/raw/` (IMD rainfall and INDmet temperature). Processed and cleaned datasets live in `data/processed/`.

---

## Benchmarking & Validation

**File:** `src/benchmark.py`

The model is rigorously validated against **21 curated historical Indian extreme weather events** spanning droughts, floods, heatwaves, and calm baseline controls.

### Benchmark Events Include

| Category | Events |
|----------|--------|
| **Drought** | Bundelkhand 2015, Banda 2015, Marathwada Latur 2015, Aurangabad 2015, Rayalaseema Anantapur 2016 |
| **Flood** | Kerala Ernakulam 2018, Kerala Idukki 2018, Kerala Wayanad 2018, Chennai 2015, Kolhapur 2019, Sangli 2019 |
| **Heatwave** | Phalodi 2016 (national record 51°C), Churu 2016, Titlagarh 2016, Nagpur 2015, Ahmedabad 2016 |
| **Controls** | Punjab Spring 2021, Pune Post-Monsoon 2022, Varanasi Autumn 2020, Hassan Pre-Monsoon 2022 |

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Disaster Recall** | ≥ 85% | Known disasters correctly classified as HIGH or SEVERE |
| **Control Specificity** | ≥ 85% | Calm periods correctly classified as LOW or MODERATE |
| **False Alarm Rate** | ≤ 15% | False disaster alarms on control cases |
| **Score Monotonicity** | 100% pass | Risk scores never decrease as stress indicators escalate |
| **Inference Latency** | < 50 ms | End-to-end pipeline latency per query |

The benchmark suite also verifies **physical monotonicity** — ensuring that as rainfall deficits deepen, precipitation intensifies, or temperatures climb, the corresponding risk scores monotonically increase.

Run the benchmark:
```bash
python src/benchmark.py
```

This produces both a terminal report and a Markdown report saved to `data/processed/benchmark_report.md`.

---

## Installation

### Prerequisites
- Python 3.10 or higher

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd environment-risk-model

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` ≥ 1.26.0 | Numerical computations |
| `pandas` ≥ 2.0.0 | Data manipulation and time-series processing |
| `scipy` ≥ 1.11.0 | KDTree for geographic resolution |
| `requests` ≥ 2.31.0 | Data fetching utilities |
| `tabulate` ≥ 0.9.0 | Table formatting |
| `rich` ≥ 13.0.0 | Enhanced terminal output |

---

## Usage

### Interactive Demo (Recommended for Presentations)

```bash
# Run automated preset case studies — ideal for pitch demos
python src/demo_predict.py --demo
```

This cycles through four landmark case studies (Marathwada drought, Kerala floods, Phalodi heatwave, and a calm baseline) with a polished formatted output.

### Query Specific Coordinates

```bash
# Evaluate risk for a specific location and date
python src/demo_predict.py --lat 23.52 --lon 87.31 --date 2023-09-19
```

### Interactive Mode

```bash
# Launch interactive CLI prompting for coordinates and date
python src/demo_predict.py
```

### Sample Output

```
================================================================================
                    AGRINODE AI -- ENVIRONMENTAL RISK MODEL
                    Case Study: Marathwada 2015 Severe Drought
================================================================================

 [LOCATION DETAILS]
    Nearest District : Latur, Maharashtra
    Coordinates      : Lat 18.4088, Lon 76.5604
    Evaluation Date  : 2015-08-30
    Inference Time   : 12.3 ms

--------------------------------------------------------------------------------
 [MULTI-HAZARD RISK ASSESSMENT]
    * Drought Risk       : SEVERE     [Score: 89.2 / 100]
    * Flood Risk Proxy   : LOW        [Score:  8.4 / 100]
    * Heat Stress        : MODERATE   [Score: 42.1 / 100]

    >> OVERALL ENVIRONMENTAL RISK: SEVERE [Score: 46.6 / 100]

--------------------------------------------------------------------------------
 [TRANSPARENT EXPLAINABILITY ("WHY?")]
    1. 90-day cumulative rainfall is 62% below local historical normal.
    2. 30-day cumulative rainfall is 58% below local historical monthly normal.
    3. Extended dry spell detected (14 consecutive days with rainfall < 2.5mm).

--------------------------------------------------------------------------------
 [OPERATIONAL FARM DECISIONS (FIELD ORDERS)]
    [IRRIGATION-CRITICAL] : IMMEDIATE: Trigger deficit drip irrigation cycle...
    [FIELD_OPS-CRITICAL]  : Apply organic mulch to reduce evapotranspiration.
    [FIELD_OPS-CRITICAL]  : Suspend ALL non-essential fertilizer applications.
================================================================================
```

### Run Benchmarks

```bash
# Execute the full ground-truth benchmark evaluation suite
python src/benchmark.py
```

### Run EDA (Exploratory Data Analysis)

```bash
# Generate EDA report on the processed datasets
python scripts/run_eda.py
```

---

## Project Structure

```
environment-risk-model/
│
├── data/
│   ├── raw/
│   │   ├── imd/                          # IMD daily district rainfall data
│   │   └── indmet/                       # INDmet daily district temperature data
│   ├── processed/
│   │   ├── district_centroids.csv        # 700+ Indian district coordinates
│   │   ├── district_baselines.csv        # Monthly climate normals & percentiles
│   │   ├── rainfall_clean.csv            # Preprocessed rainfall time series
│   │   ├── temperature_clean.csv         # Preprocessed temperature time series
│   │   ├── benchmark_report.md           # Generated benchmark report
│   │   ├── eda_report.md                 # Generated EDA report
│   │   └── eda_summary.json              # EDA statistics
│   └── benchmark_cases.csv              # 21 curated ground-truth events
│
├── src/
│   ├── __init__.py
│   ├── geo_resolver.py                   # KDTree coordinate-to-district resolver
│   ├── data_loader.py                    # Raw data ingestion & validation
│   ├── preprocessing.py                  # Data cleaning & date harmonization
│   ├── baseline.py                       # District historical baseline computation
│   ├── features.py                       # Multi-scale feature extraction engine
│   ├── drought.py                        # Drought risk scoring engine
│   ├── flood.py                          # Flood proxy risk scoring engine
│   ├── heat.py                           # Heat stress risk scoring engine
│   ├── overall_risk.py                   # Multi-hazard severity synthesis
│   ├── decision_engine.py                # Farm decision & advisory generation
│   ├── benchmark.py                      # Ground-truth benchmark evaluation suite
│   └── demo_predict.py                   # Interactive CLI demo tool
│
├── scripts/
│   ├── build_district_centroids.py       # Compile district centroid registry
│   ├── run_eda.py                        # Exploratory data analysis pipeline
│   ├── evaluate_phase4.py                # Phase 4 risk engine evaluation
│   └── inspect_benchmark_features.py     # Feature inspection for benchmark cases
│
├── tests/
│   ├── test_phase2.py                    # Preprocessing & geo resolver tests
│   ├── test_phase3.py                    # Baseline & feature engineering tests
│   ├── test_phase4.py                    # Risk engine calibration tests
│   ├── test_phase5.py                    # Decision engine tests
│   └── test_phase6.py                    # Benchmark evaluation tests
│
├── Architecture.md                       # Detailed system architecture blueprint
├── Phase_Wise_Plan.md                    # Phase-wise implementation plan
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## Testing

The project includes a comprehensive test suite covering all pipeline phases:

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific phase
pytest tests/test_phase2.py -v    # Preprocessing & Geographic Resolution
pytest tests/test_phase3.py -v    # Baseline Calibration & Feature Engineering
pytest tests/test_phase4.py -v    # Risk Engine Calibration & Scoring
pytest tests/test_phase5.py -v    # Farm Decision Engine
pytest tests/test_phase6.py -v    # Benchmark Evaluation Suite
```

### What the Tests Cover

- **Phase 2:** Data cleaning, schema standardization, date harmonization, KDTree geo-resolution accuracy, and missing value handling.
- **Phase 3:** Baseline statistical computation, multi-scale rolling feature extraction, anomaly calculation correctness, and dry/hot spell detection.
- **Phase 4:** Individual risk engine scoring, tier boundary validation, score clamping (0–100), physical monotonicity, and multi-hazard synthesis logic.
- **Phase 5:** Action matrix completeness, priority ranking, de-duplication of routine actions, advisory summary generation, and decision plan structure.
- **Phase 6:** End-to-end benchmark pipeline execution, SLA metric computation, recall/specificity thresholds, and report generation.

---

## Future Scope & Enhancements

### Real-Time Data Integration
- **Live Weather API Feeds:** Integrate real-time weather data from OpenWeatherMap, Open-Meteo, or IMD APIs to enable current-day and forecast-based risk assessments instead of relying solely on historical records.
- **Satellite Remote Sensing:** Incorporate NDVI (vegetation index) and soil moisture data from ISRO/Sentinel satellites for real-time crop health monitoring.

### Enhanced Risk Modeling
- **Crop-Specific Risk Profiles:** Tailor risk thresholds based on crop type (rice, wheat, cotton, sugarcane) since different crops have different vulnerability windows and stress tolerances.
- **Soil Type Integration:** Factor in soil water-holding capacity, drainage characteristics, and soil texture to improve drought and flood risk accuracy.
- **Wind & Storm Modeling:** Add a cyclone/wind damage risk engine using wind speed data, particularly for coastal agricultural regions.
- **Pest & Disease Correlation:** Map environmental conditions (prolonged humidity, post-flood moisture) to pest/disease outbreak probability.

### Machine Learning Upgrades
- **ML-Based Calibration:** Train gradient-boosted or neural network models on labeled disaster data to learn optimal feature weights instead of using manually tuned thresholds.
- **Ensemble Risk Scoring:** Combine rule-based engines with ML predictions for hybrid risk assessment with improved accuracy.
- **Temporal Forecasting:** Implement LSTM or transformer-based time-series models for 7-day and 14-day advance risk forecasting.

### Platform & Integration
- **REST API Layer:** Expose the risk assessment pipeline as a FastAPI/Flask web service for integration with mobile apps, dashboards, and IoT systems.
- **SMS/WhatsApp Alerts:** Push critical-priority farm advisories directly to farmers' phones in regional languages.
- **Dashboard Visualization:** Build an interactive web dashboard with maps, risk heatmaps, and trend charts using Plotly/Dash or Streamlit.
- **IoT Sensor Fusion:** Integrate with on-farm IoT sensors (soil moisture probes, weather stations) for hyperlocal field-level risk assessment.

### Geographic Expansion
- **Block/Taluk-Level Resolution:** Move from district-level to sub-district (block/taluk) granularity for more precise localized risk.
- **Multi-Country Support:** Extend the model to support other South Asian and tropical agriculture regions with similar monsoon-dependent farming.

### Operational Enhancements
- **Insurance Risk Scoring:** Generate standardized risk scores compatible with crop insurance frameworks (PMFBY) for automated claim assessment.
- **Supply Chain Impact Analysis:** Predict how environmental risks at farm level cascade into regional supply chain disruptions.
- **Historical Trend Analysis:** Generate long-term climate risk trend reports for districts to support agricultural policy planning.

---

## License

This project is part of the **AgriNode AI** initiative. Please refer to the project's license terms for usage and distribution policies.
