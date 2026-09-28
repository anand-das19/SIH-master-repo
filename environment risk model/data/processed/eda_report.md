# AgriNode AI — Phase 1 EDA & Climate Profiling Report

**Dataset:** INDmet High-Resolution Daily District Climate (1981-2024)
**Coverage:** 1981 - 2024 (44 Years) | **Districts:** 726

## 1. Data Quality & Integrity
- **Data Completeness:** `100.0%`
- **Missing Value Count:** `0`
- **Sentinel Values (-999.9):** `0`

## 2. Representative District Distributions

| ID | District | State | Max Rain (mm) | Rain 95th %ile | Mean Tmax (°C) | Max Tmax (°C) |
|---|---|---|---|---|---|---|
| 14 | Agra | Uttar Pradesh | 122.93 | 12.78 | 31.65 | 46.51 |
| 16 | Alappuzha | Kerala | 166.92 | 34.09 | 30.62 | 35.08 |
| 25 | Anantapur | Andhra Pradesh | 59.16 | 8.48 | 32.11 | 41.22 |
| 47 | Banda | Uttar Pradesh | 93.55 | 15.84 | 31.61 | 46.43 |
| 70 | Bijnor | Uttar Pradesh | 265.39 | 21.71 | 28.69 | 44.7 |
| 122 | Dhenkanal | Odisha | 78.93 | 22.72 | 32.02 | 42.76 |
| 148 | Gir Somnath | Gujarat | 288.51 | 11.52 | 31.37 | 40.26 |
| 205 | Kasganj | Uttar Pradesh | 219.61 | 14.42 | 30.94 | 45.47 |
| 231 | Kushinagar | Uttar Pradesh | 216.04 | 24.78 | 31.05 | 43.29 |
| 274 | North Goa | Goa | 321.74 | 41.88 | 32.21 | 37.88 |
| 425 | Kishtwar | Jammu And Kashmir | 141.43 | 8.64 | 19.12 | 33.96 |
| 500 | Jalna | Maharashtra | 158.24 | 13.39 | 31.72 | 43.28 |

## 3. Key Meteorological Insights for Risk Modeling
- **Rainfall Asymmetry:** Across arid regions (e.g. Rajasthan, Rayalaseema), dry days (>90% of year) dominate, requiring robust baseline normalization (Phase 3).
- **Extreme Runoff Spikes:** High-range coastal districts (e.g., Kerala, Western Ghats) experience extreme single-day spikes >300 mm, validating the need for acute 95th/99th percentile flood thresholds.
- **Thermal Departure Consistency:** Maximum summer temperatures routinely exceed 45°C in northern/central plains, confirming the necessity of localized anomaly computation rather than fixed nationwide thresholds.
