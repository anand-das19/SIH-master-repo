from src.features import get_feature_extractor
import pandas as pd

ext = get_feature_extractor()
df = pd.read_csv('data/benchmark_cases.csv')
for idx, row in df.iterrows():
    f = ext.extract_features(lat=row['latitude'], lon=row['longitude'], query_date=row['query_date'])
    print(f"{row['case_id']:<12} {row['category']:<9} {row['district']:<12} | "
          f"r1d={f['rain_1d']:>5.1f} r7d={f['rain_7d']:>5.1f} r30d={f['rain_30d']:>6.1f} r90d={f['rain_90d']:>6.1f} "
          f"a30={f['anomaly_30d']:>6.1f}% a90={f['anomaly_90d']:>6.1f}% dry={f['current_dry_spell']:>2d}d | "
          f"tmax={f['tmax']:>4.1f} tanom={f['tmax_anomaly']:>4.1f} chot={f['consecutive_hot_days']:>2d}d | "
          f"p90={f['baseline_rain_p90']:>4.1f} p95={f['baseline_rain_p95']:>4.1f} tp90={f['baseline_tmax_p90']:>4.1f}")
