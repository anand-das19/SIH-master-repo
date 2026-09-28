# AgriNode AI — Environmental Risk Monitoring
## Detailed Phase-Wise Implementation, Benchmarking & Demo Plan (V1)

**Objective:** Build, calibrate, empirically benchmark, and demonstrate a working environmental risk intelligence model prototype for the SIH/PPT screening stage.

**Core Idea:**

> **Latitude + Longitude + Date + Historical Baseline → Calibrated Multi-Hazard Risk Engines → Farm Decision Engine → Empirical Benchmarking → Judge-Ready CLI Demo**

---

## Architectural Principles & Alignment

Taking inspiration from our Smart Irrigation ML/DL pipeline:
1. **Clear Modular Phases:** Single responsibilities, deterministic inputs, measurable outputs, and testable deliverables.
2. **Empirical Benchmarking First:** Models must be tested against documented ground-truth extreme events and controls *before* presenting the demo.
3. **From Risk Scores to Farm Decisions:** Models do not stop at abstract risk scores; a dedicated Decision Engine translates risk classes into concrete, operational field commands.
4. **Judge-Ready Interactive Demo:** The final deliverable is `src/demo_predict.py`, an interactive CLI with built-in preset case studies for flawless pitch presentations.
5. **No Web API / Frontend Bloat:** Zero dependencies on FastAPI, web servers, or complex UI frameworks for V1.

---

# Phase 1 — Data Ingestion & Exploratory Data Analysis (EDA)

## Single Responsibility
**Ingest the historical weather datasets, analyze climate distributions, and catalog historical ground-truth benchmark events.**

### Tasks
1. **Acquire Primary Datasets:**
   - Ingest IMD Daily District Rainfall (NWDP / IMD Archive).
   - Ingest INDmet Daily District Temperature (Zenodo 1981–2024, ~153.2 MB package).
   - *Fallback:* Curated open IMD district rainfall dataset / Open-Meteo historical archive.
2. **Exploratory Data Analysis (EDA):**
   - Inspect distribution of daily rainfall, extreme wet spells, dry spells, and Tmax percentiles.
   - Profile missing values and sentinel codes (e.g. `-999.9`).
3. **Compile Ground-Truth Benchmark Catalog (`data/benchmark_cases.csv`):**
   - Curate 15–20 documented real-world historical Indian disaster events across 4 categories:
     - **Drought:** Bundelkhand (2015), Marathwada/Latur (2015), Rayalaseema (2016).
     - **Excess Rain / Flood:** Kerala Floods (Aug 2018), Chennai Deluge (Dec 2015), Kolhapur/Sangli (Aug 2019).
     - **Heatwave:** Churu/Phalodi (May 2016), Titlagarh (April 2016), Nagpur (May 2015).
     - **Normal Controls:** Balanced baseline weeks across distinct agro-climatic zones to verify low false alarm rates.
4. **Compile District Coordinates (`data/processed/district_centroids.csv`):**
   - Reference table of Indian district names, states, and centroid latitudes/longitudes.

### Deliverables
- `data/raw/imd/` & `data/raw/indmet/`
- `data/benchmark_cases.csv`
- `data/processed/district_centroids.csv`

---

# Phase 2 — Data Preprocessing & Geographic Resolution

## Single Responsibility
**Clean and harmonize time-series datasets and implement instant coordinate-to-district resolution.**

### Tasks
1. **Standardize Schema & Timestamps:**
   - Align column names: `date`, `state`, `district`, `rainfall`, `tmax`, `tmin`.
   - Convert all date strings to standard `YYYY-MM-DD`.
2. **Harmonize District Names:**
   - Map spelling variations between IMD and INDmet (e.g., "Burdwan" vs. "Purba Bardhaman").
3. **Handle Missing Values:**
   - Impute short gaps ($< 3\text{ days}$) via seasonal interpolation; do not blindly overwrite missing rainfall with zeros.
4. **Build KDTree Geographic Resolver (`src/geo_resolver.py`):**
   - Load `district_centroids.csv` into `scipy.spatial.KDTree`.
   - Implement `resolve_district(lat, lon)` returning the nearest valid district and state in $<1\text{ ms}$.
   - Guarantees zero complex GIS/GDAL dependency installation issues on Windows.
5. **Temporal Split (Calibration vs. Validation):**
   - Split historical records into Calibration/Baseline Period (1981–2015) and Validation/Testing Period (2016–2024) to prevent lookahead bias during feature validation.

### Deliverables
- `data/processed/rainfall_clean.csv`
- `data/processed/temperature_clean.csv`
- `src/geo_resolver.py`

---

# Phase 3 — Baseline Calibration & Feature Engineering

## Single Responsibility
**Compute local historical climate baselines and generate multi-scale rolling feature vectors.**

### Tasks
1. **Compute District Historical Baselines (`data/processed/district_baselines.csv`):**
   - For every district and calendar month, pre-compute:
     - $\mu_{\text{rain\_month}}, \sigma_{\text{rain\_month}}$ (monthly precipitation normal & variance)
     - $\mu_{\text{tmax\_month}}, \sigma_{\text{tmax\_month}}$ (monthly temperature normal & variance)
     - 90th & 95th historical daily rainfall percentiles (flash-flood thresholds)
     - 90th historical daily Tmax percentile (local heatwave threshold)
2. **Engineer Multi-Scale Feature Matrix (`src/features.py`):**
   - **Rainfall Indicators:**
     - Short-term: `rain_1d`, `rain_7d`
     - Medium/Long-term: `rain_30d`, `rain_90d`
     - Anomaly / Deficit: $\text{anomaly\_30d} = \frac{\text{rain\_30d} - \text{baseline\_30d}}{\text{baseline\_30d}} \times 100$
     - Dryness: `current_dry_spell` (consecutive days with rain $< 2.5\text{ mm}$)
     - Saturation: `max_rainfall_1d_30d`, `heavy_rain_days_7d`
   - **Thermal Indicators:**
     - `tmax`, `tmin`, `mean_temp`
     - `tmax_anomaly` (departure in °C from monthly historical mean)
     - `consecutive_hot_days` (days exceeding local 90th percentile)
     - `max_tmax_7d`

### Deliverables
- `data/processed/district_baselines.csv`
- `src/features.py`

---

# Phase 4 — Risk Engines & Multi-Hazard Calibration

## Single Responsibility
**Implement and calibrate the three hazard models and aggregate them into an overall environmental risk rating.**

### Standardized Risk Tiers
Every hazard engine maps continuous scores ($0–100$) into 4 standardized tiers:
- `0 – 30` : **LOW**
- `31 – 60` : **MODERATE**
- `61 – 80` : **HIGH**
- `81 – 100`: **SEVERE**

### Engine Implementations
1. **Drought Risk Engine (`src/drought.py`):**
   - Multi-scale deficit scoring: $40\%$ long-term anomaly $+ 30\%$ medium-term anomaly $+ 30\%$ dry-spell persistence.
   - Returns risk score, risk tier, and human-readable factors.
2. **Flood Risk Proxy Engine (`src/flood.py`):**
   - Acute precipitation intensity: $50\%$ peak daily rainfall vs historical percentiles $+ 30\%$ 7-day accumulation $+ 20\%$ heavy-rain frequency.
   - Explicitly labeled **Flood Risk Proxy** (soil saturation & excess surface runoff index).
3. **Heat Stress Engine (`src/heat.py`):**
   - Thermal stress index: $40\%$ local baseline departure $+ 30\%$ absolute thermal threshold $+ 30\%$ heatwave persistence.
4. **Multi-Hazard Synthesis Engine (`src/overall_risk.py`):**
   - Aggregates multi-hazard risk with severity dominance:
     ```python
     if any(risk == "SEVERE" for risk in [drought, flood, heat]):
         overall = "SEVERE"
     elif [drought, flood, heat].count("HIGH") >= 2 or any(risk == "HIGH" for risk in [drought, flood, heat]):
         overall = "HIGH"
     elif any(risk == "MODERATE" for risk in [drought, flood, heat]):
         overall = "MODERATE"
     else:
         overall = "LOW"
     ```

### Deliverables
- `src/drought.py`
- `src/flood.py`
- `src/heat.py`
- `src/overall_risk.py`

---

# Phase 5 — Farm Decision Engine (`src/decision_engine.py`)

## Single Responsibility
**Translate abstract risk categories into concrete, operational field commands for farmers.**

Just as the Smart Irrigation model converts Class 1 predictions into explicit pump runtimes (e.g. 25 mins), AgriNode converts environmental risk into actionable agronomic decisions:

### Action Matrix
- **Drought `HIGH` / `SEVERE`:**
  - *Irrigation Order:* Trigger deficit drip irrigation cycle; target root-zone replenishment.
  - *Field Operations:* Apply organic mulch to reduce evapotranspiration; suspend non-essential fertilizer applications.
- **Flood `HIGH` / `SEVERE`:**
  - *Irrigation Order:* Immediate shutdown of all pumps and irrigation lines.
  - *Field Operations:* Open drainage channels; clear field furrows to prevent waterlogging; prepare preventative anti-fungal spray post-recession.
- **Heat `HIGH` / `SEVERE`:**
  - *Irrigation Order:* Schedule early-morning light canopy cooling cycle.
  - *Field Operations:* Shift all manual farm labor to before 10:00 AM; avoid mid-day chemical spraying to prevent leaf scorching.
- **Low / Moderate Risk:**
  - Normal routine monitoring and scheduled seasonal maintenance.

### Deliverable
- `src/decision_engine.py`

---

# Phase 6 — Rigorous Testing & Ground-Truth Benchmarking (`src/benchmark.py`)

## Single Responsibility
**Empirically evaluate and benchmark the entire model pipeline against historical disaster ground-truth events BEFORE building the final demo.**

### Benchmark Evaluation Suite
Run automatically via:
```bash
python src/benchmark.py
```

### Evaluation Protocol
1. Load `data/benchmark_cases.csv` (15–20 curated events: historical droughts, floods, heatwaves, and seasonal controls).
2. For each case, resolve coordinates, extract features, execute the 3 risk engines, synthesize overall risk, and verify decisions.
3. Compare model outputs against ground-truth labels.

### Target Quantitative Benchmarking Metrics:
| Metric | Benchmark Target | Description |
|---|---|---|
| **Disaster Recall** | $\ge 85\%$ | Correct detection of known historical disasters as `HIGH` or `SEVERE` |
| **Control Specificity** | $\ge 85\%$ | Correct classification of normal baseline periods as `LOW` or `MODERATE` |
| **False Alarm Rate** | $\le 15\%$ | False disaster alarms on calm seasonal control cases |
| **Score Monotonicity** | $100\%$ pass | As rainfall deficit or Tmax increases, risk score must not decrease |
| **Inference Latency** | $< 50\text{ ms}$ | Sub-second runtime per coordinate query for seamless live pitch demo |

### Output Deliverable
- `src/benchmark.py`
- Automated terminal & Markdown Benchmark Report (perfect for the SIH PPT presentation slides!).

---

# Phase 7 — Final Interactive Judge-Ready Demo (`src/demo_predict.py`)

## Single Responsibility
**Provide a polished, judge-ready CLI demonstration tool built directly on the benchmarked and validated pipeline.**

Executed **after** benchmarking is verified and locked in.

### Execution Modes
```bash
# Mode 1: Automated Preset Case Studies (Ideal for Pitch Presentation)
python src/demo_predict.py --demo

# Mode 2: Specific Coordinate & Date Testing
python src/demo_predict.py --lat 23.52 --lon 87.31 --date 2023-09-19

# Mode 3: Interactive CLI Prompting
python src/demo_predict.py
```

### Presentation Output Format
```text
================================================================================
                    AGRINODE AI — ENVIRONMENTAL RISK MODEL
================================================================================

📍 LOCATION DETAILS
   Nearest District : Durgapur, West Bengal
   Coordinates      : Lat 23.5204, Lon 87.3119
   Evaluation Date  : 2023-09-19

--------------------------------------------------------------------------------
📊 MULTI-HAZARD RISK ASSESSMENT
   • Drought Risk       : HIGH       [Score: 74 / 100]
   • Flood Risk Proxy   : LOW        [Score: 16 / 100]
   • Heat Stress        : MODERATE   [Score: 54 / 100]

   🚨 OVERALL ENVIRONMENTAL RISK: HIGH [Score: 74 / 100]

--------------------------------------------------------------------------------
🔍 TRANSPARENT EXPLAINABILITY ("WHY?")
   1. 30-day cumulative rainfall is 41.2% below local historical monthly normal.
   2. Extended dry spell detected (9 consecutive days with rainfall < 2.5mm).
   3. Current Tmax (35.4°C) is 2.9°C above historical monthly baseline.

--------------------------------------------------------------------------------
🚜 OPERATIONAL FARM DECISIONS (FIELD ORDERS)
   [IRRIGATION] : Trigger supplemental deficit irrigation; target root-zone moisture.
   [FIELD WORK] : Apply surface mulching to curb evaporative water loss.
   [ADVISORY]   : Postpone scheduled nitrogenous fertilizer application until soil moisture recovers.
================================================================================
```

### Deliverable
- `src/demo_predict.py`

---

# Complete Execution Roadmap

```text
Phase 1: Ingestion & EDA          (IMD + INDmet + Benchmark Cases)
   │
   ▼
Phase 2: Preprocessing & Geo       (Clean data + KDTree resolver)
   │
   ▼
Phase 3: Baselines & Features      (District normals + Rolling indicators)
   │
   ▼
Phase 4: Calibrated Risk Engines   (Drought + Flood + Heat + Overall)
   │
   ▼
Phase 5: Farm Decision Engine      (Concrete agronomic field commands)
   │
   ▼
Phase 6: Ground-Truth Benchmarks   (src/benchmark.py -> Metrics & Recall Report)
   │
   ▼
Phase 7: Judge-Ready Demo CLI      (src/demo_predict.py -> Live Pitch Demonstration)
```

---

# Final Deliverables Map

| Phase | Milestone | File Path | Deliverable Function |
|---|---|---|---|
| **Phase 1** | Ingestion & EDA | `data/benchmark_cases.csv` | Dataset catalogs & ground-truth events |
| **Phase 2** | Geo Resolution | `src/geo_resolver.py` | KDTree coordinate resolver ($<1\text{ ms}$) |
| **Phase 3** | Baselines & Features | `src/features.py` | Rolling multi-scale feature vectors |
| **Phase 4** | Risk Engines | `src/drought.py`, `src/flood.py`, `src/heat.py`, `src/overall_risk.py` | Calibrated hazard classifiers (0–100) |
| **Phase 5** | Farm Decision Engine | `src/decision_engine.py` | Concrete field & irrigation directives |
| **Phase 6** | **Empirical Benchmarking** | **`src/benchmark.py`** | **Ground-truth validation, recall & latency report** |
| **Phase 7** | **Final Demo CLI** | **`src/demo_predict.py`** | **Judge-ready presentation CLI with presets** |
