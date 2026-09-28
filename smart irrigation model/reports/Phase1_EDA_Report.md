# Phase 1: Exploratory Data Analysis Report

## 1. Executive Summary
- Dataset: dataset/data.csv
- Total records: 16411 rows (zero null values)
- Class 0 (Pump OFF): 9062 (55.22%)
- Class 1 (Pump ON): 7349 (44.78%)

## 2. Correlations with Target
- MOI: -0.1873
- temp: 0.5842
- humidity: -0.5381

## 3. Generated Visualizations
Saved in reports/figures:
- 01_class_distribution.png
- 02_feature_boxplots.png
- 03_correlation_matrix.png
- 04_temp_humidity_scatter.png
- 05_moi_density.png

## 4. Train / Val / Test Splits Created
- dataset/train.csv (11,487 rows, 70 percent)
- dataset/val.csv (2,462 rows, 15 percent)
- dataset/test.csv (2,462 rows, 15 percent)
