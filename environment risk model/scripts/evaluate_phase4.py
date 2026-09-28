import pandas as pd
from src.features import get_feature_extractor
from src.overall_risk import compute_overall_risk

ext = get_feature_extractor()
df = pd.read_csv('data/benchmark_cases.csv')

correct = 0
total = len(df)
print("=" * 80)
print("AGRINODE AI - PHASE 4: RISK ENGINES CALIBRATION BENCHMARK EVALUATION")
print("=" * 80)

disaster_hits = 0
disaster_total = 0
control_hits = 0
control_total = 0

for _, row in df.iterrows():
    f = ext.extract_features(lat=row['latitude'], lon=row['longitude'], query_date=row['query_date'])
    r = compute_overall_risk(f)
    overall = r['overall_tier']
    expected = row['expected_tier']
    cat = row['category']
    
    if cat == 'CONTROL':
        control_total += 1
        hit = overall in ('LOW', 'MODERATE')
        if hit:
            control_hits += 1
    else:
        disaster_total += 1
        hit = overall in ('HIGH', 'SEVERE')
        if hit:
            disaster_hits += 1
            
    if hit:
        correct += 1
        
    res = "[PASS]" if hit else "[FAIL]"
    d_score = r['drought']['drought_score']
    f_score = r['flood']['flood_score']
    h_score = r['heat']['heat_score']
    print(f"{row['case_id']:<12} | {cat:<9} | exp={expected:<7} got={overall:<8} {res} | D={d_score:>4.1f} F={f_score:>4.1f} H={h_score:>4.1f} | Dominant: {r['primary_hazard']}")

print("-" * 80)
print(f"Disaster Recall   : {disaster_hits}/{disaster_total} ({disaster_hits/disaster_total*100:.1f}%) [Target >= 85%]")
print(f"Control Specificity: {control_hits}/{control_total} ({control_hits/control_total*100:.1f}%) [Target >= 85%]")
print(f"Overall Accuracy   : {correct}/{total} ({correct/total*100:.1f}%)")
print("=" * 80)
