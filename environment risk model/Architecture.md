# AgriNode AI — Environmental Risk Monitoring
## System Architecture & Engineering Blueprint (V1)

**V1 Goal:** Build an explainable, empirically benchmarked, and judge-ready environmental risk model prototype for the SIH/PPT screening stage.

**Core Idea:**

> **Latitude + Longitude + Date + Historical Climate Baseline → Calibrated Risk Engines (Drought, Flood Proxy, Heat Stress) → Multi-Hazard Synthesis → Agronomic Decision Engine → Ground-Truth Benchmarking → Judge-Ready CLI Demo**

---

# 1. System Pipeline Architecture

```text
                                  INPUT
                       Latitude, Longitude, Date
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Geographic Resolver   │
                       │  (src/geo_resolver.py)  │
                       │   KDTree Centroid Match │
                       └────────────┬────────────┘
                                    │ District, State
                                    ▼
                       ┌─────────────────────────┐
                       │  Historical Baseline &  │
                       │   Feature Engineering   │
                       │    (src/features.py)    │
                       │ Rolling Windows & Z-Norm│
                       └────────────┬────────────┘
                                    │ Feature Vector
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
  │   DROUGHT    │           │ FLOOD PROXY  │           │ HEAT STRESS  │
  │    ENGINE    │           │    ENGINE    │           │    ENGINE    │
  │(src/drought) │           │ (src/flood)  │           │  (src/heat)  │
  └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │   Multi-Hazard Engine   │
                       │  (src/overall_risk.py)  │
                       │ Calibrated Risk Levels  │
                       └────────────┬────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │  Farm Decision Engine   │
                       │ (src/decision_engine.py)│
                       │ Actionable Field Orders │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                     │
         ▼ STEP 1 (VERIFICATION & VALIDATION)                  ▼ STEP 2 (FINAL DEMONSTRATION)
┌─────────────────────────────────┐                   ┌─────────────────────────────────┐
│        src/benchmark.py         │                   │       src/demo_predict.py       │
│  Empirical Testing Suite        │                   │   Judge-Ready Interactive CLI   │
│  • 15+ Disaster Benchmarks      │                   │   • Preset Case Studies         │
│  • Recall, Specificity, Latency │                   │   • Interactive Lat/Lon Input   │
│  • Quantitative Report for PPT  │                   │   • Actionable Advisory Output  │
└─────────────────────────────────┘                   └─────────────────────────────────┘
```

---

# 2. Datasets & Ingestion Strategy

### 1. IMD Daily District Rainfall (NWDP / IMD Archive)
- **Role:** Primary precipitation source for Drought and Flood Risk Proxy.
- **Fields:** Daily rainfall (`mm`), normal rainfall, cumulative precipitation.
- *Fallback:* Curated open IMD historical district tables / Open-Meteo historical archive.

### 2. INDmet Daily District Temperature (Zenodo 1981–2024)
- **Role:** Temperature records for Heat Stress modeling.
- **Fields:** `Tmax` (°C), `Tmin` (°C), `Tmean` (°C).
- *Size:* Lightweight district-data package (~153.2 MB).

---

# 3. Geographic Resolution (No Complex GIS)

To prevent platform/OS install failures (avoiding GDAL / GeoPandas):
- Pre-compiled district coordinate registry: `data/processed/district_centroids.csv`.
- `src/geo_resolver.py` utilizes `scipy.spatial.KDTree` to map user coordinates to the nearest district centroid in $<1\text{ ms}$.

---

# 4. Feature Engineering & Historical Baselines

Risk is calculated strictly relative to local historical norms:
- **Baseline Calibration:** District-month normals ($\mu_{\text{rain}}, \sigma_{\text{rain}}, \mu_{\text{tmax}}, \sigma_{\text{tmax}}$, 90th percentiles) pre-computed in `data/processed/district_baselines.csv`.
- **Rainfall Features:** `rain_1d`, `rain_7d`, `rain_30d`, `rain_90d`, `rainfall_anomaly_30d` (%), `current_dry_spell` (days), `heavy_rain_days_7d`.
- **Temperature Features:** `tmax`, `tmin`, `tmax_anomaly` (°C above local monthly normal), `consecutive_hot_days` ($>90\text{th}$ percentile), `max_tmax_7d`.

---

# 5. Calibrated Risk Engines

All risk engines output a continuous score ($0–100$) and a discrete tier:
`LOW` (0–30), `MODERATE` (31–60), `HIGH` (61–80), `SEVERE` (81–100).

1. **Drought Engine (`src/drought.py`):**
   - Weighted multi-scale deficit: $40\%$ 90-day anomaly $+ 30\%$ 30-day anomaly $+ 30\%$ dry-spell severity.
2. **Flood Proxy Engine (`src/flood.py`):**
   - Acute precipitation extremeness: $50\%$ 1-day peak vs historical 95th/99th percentiles $+ 30\%$ 7-day soil saturation $+ 20\%$ heavy-rain frequency.
3. **Heat Stress Engine (`src/heat.py`):**
   - Thermal anomaly & persistence: $40\%$ deviation above monthly normal $+ 30\%$ absolute thermal threshold $+ 30\%$ consecutive heatwave days.
4. **Multi-Hazard Synthesis (`src/overall_risk.py`):**
   - Synthesizes multi-hazard exposure into an aggregate environmental risk rating.

---

# 6. Farm Decision Engine (`src/decision_engine.py`)

Rather than only presenting raw risk numbers, the model outputs **concrete, operational farm decisions**:
- **Drought `HIGH`/`SEVERE`:** Trigger emergency deficit irrigation; recommend mulching and soil moisture retention; postpone chemical spraying.
- **Flood `HIGH`/`SEVERE`:** Immediately halt all irrigation; inspect drainage ditches and low-lying furrows; prepare for post-waterlogging fungal treatments.
- **Heat `HIGH`/`SEVERE`:** Schedule early-morning/late-evening farm operations; avoid mid-day pesticide application; execute protective light canopy misting.
- **Normal / Low Risk:** Standard seasonal field maintenance.

---

# 7. Model Testing & Ground-Truth Benchmarking (`src/benchmark.py`)

Executed **before** presenting the demo to ensure empirical rigor:
- Evaluates against `data/benchmark_cases.csv` (15–20 curated ground-truth historical Indian extreme events):
  - Droughts: Bundelkhand 2015, Marathwada 2015, Rayalaseema 2016.
  - Floods / Deluges: Kerala Aug 2018, Chennai Dec 2015, Kolhapur Aug 2019.
  - Heatwaves: Churu May 2016, Titlagarh April 2016, Nagpur May 2015.
  - Controls: Calm seasonal baseline weeks across 4 distinct agro-climatic zones.
- **Quantitative Metrics Produced:**
  - Extreme Event Recall ($\ge 85\%$)
  - False Alarm Rate on Controls ($\le 15\%$)
  - Score Monotonicity and Boundary Invariance ($0 \le \text{score} \le 100$)
  - Inference Latency ($< 50\text{ ms}$)

---

# 8. Final Interactive Demo CLI (`src/demo_predict.py`)

The final user-facing deliverable built on top of the tested and verified model:
- **Preset Demonstration Scenarios (`--demo`):** Instant 1-click runs showcasing benchmark drought, flood, and heatwave scenarios for pitch presentations.
- **Interactive Mode:** User can enter any real-world Indian coordinates and historical query date.
- **Rich Terminal Presentation:** Clean ASCII risk dashboard, transparent explainability factors ("Why?"), and concrete agronomic field instructions.

---

# 9. Minimal Project Structure

```text
environment-risk/
│
├── data/
│   ├── raw/
│   │   ├── imd/
│   │   └── indmet/
│   ├── processed/
│   │   ├── district_centroids.csv
│   │   ├── district_baselines.csv
│   │   └── environmental_features.csv
│   └── benchmark_cases.csv       # Ground-truth historical disaster events
│
├── src/
│   ├── __init__.py
│   ├── geo_resolver.py           # KDTree coordinate resolution
│   ├── data_loader.py            # Raw data ingestion & validation
│   ├── preprocessing.py          # Data cleaning & date harmonization
│   ├── features.py               # Multi-scale features & baseline anomalies
│   ├── drought.py                # Drought risk engine & factors
│   ├── flood.py                  # Flood proxy engine & factors
│   ├── heat.py                   # Heat stress engine & factors
│   ├── overall_risk.py           # Multi-hazard synthesis logic
│   ├── decision_engine.py        # Farm operations & actionable advisories
│   ├── benchmark.py              # STEP 1: Empirical validation & benchmark metrics
│   └── demo_predict.py           # STEP 2: Final judge-ready interactive CLI demo
│
├── requirements.txt
└── README.md
```
