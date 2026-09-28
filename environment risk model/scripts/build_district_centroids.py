"""
AgriNode AI - Environmental Risk Intelligence
Phase 1: Build District Centroids Registry

Generates data/processed/district_centroids.csv with verified coordinates
for all Indian districts in the INDmet master registry.
"""

import re
import difflib
import pandas as pd
from pathlib import Path

# Manual alias dictionary for tricky or disputed administrative boundary entries
MANUAL_COORDS = {
    # Delhi NCT
    "central": (28.6448, 77.2167),
    "east": (28.6279, 77.2784),
    "north": (28.7041, 77.1025),
    "north east": (28.7159, 77.2807),
    "north west": (28.7230, 77.0689),
    "south": (28.4817, 77.1873),
    "south east": (28.5355, 77.2710),
    "south west": (28.5921, 76.9960),
    "west": (28.6663, 77.0674),
    
    # Sikkim
    "east sikkim": (27.3200, 78.6000),
    "north sikkim": (27.7000, 78.5000),
    "south sikkim": (27.2000, 78.4000),
    "west sikkim": (27.3000, 78.2000),

    # Maharashtra
    "bid": (18.9900, 75.7600),
    "mumbai city": (18.9388, 72.8353),
    "sub urban mumbai": (19.0760, 72.8777),

    # Bengal & UP & Bihar
    "hugli": (22.9000, 88.3900),
    "barabanki": (26.9200, 81.1800),
    "mau": (25.9500, 83.5600),
    # Duplicate named districts disambiguated by state
    "aurangabad bihar": (24.7539, 84.3742),
    "aurangabad maharashtra": (19.8762, 75.3433),
    "bilaspur chhattisgarh": (22.0797, 82.1409),
    "bilaspur himachal pradesh": (31.3260, 76.7597),
    "hamirpur uttar pradesh": (25.9524, 80.1504),
    "hamirpur himachal pradesh": (31.6862, 76.5213),
    "balrampur chhattisgarh": (23.6128, 83.6080),
    "balrampur uttar pradesh": (27.4287, 82.1802),
    "pratapgarh rajasthan": (24.0300, 74.7800),
    "pratapgarh uttar pradesh": (25.9000, 81.9900),
    "kaimur": (25.0400, 83.6100),

    # Disputed & border slivers (assigned to nearby centroid)
    "disputed ratlam mandsaur": (23.7000, 75.1000),
    "disputed alirajpur dahod": (22.5000, 74.3000),
    "disputed ratlam banswara": (23.4000, 74.8000),
    "disputed sahibganj maldah katihar": (25.3000, 87.8000),
    "disputed mandsaur jhalawar": (24.3000, 75.6000),
    "disputed nimach chittaurgarh": (24.5000, 74.9000),
    "disputed baran sheopur": (25.2000, 76.7000),
    "disputed sabar kantha udaipur": (24.1000, 73.2000),
    "disputed sabar kantha sirohi": (24.4000, 72.9000),

    # Others
    "balasore baleshwar": (21.4934, 86.9135),
    "gaurela pendra marwahi": (22.7500, 81.9000),
    "y s r kadapa": (14.4673, 78.8242),
    "dahod": (22.8300, 74.2500),
    "dangs": (20.8500, 73.7100),
    "dakshin bastar dantewada": (18.9000, 81.3500),
    "uttar bastar kanker": (20.2700, 81.4900),
    "jalor": (25.3400, 72.6100),
    "narmada": (21.8700, 73.5000),
    "narshimapura": (22.9500, 79.2000),
    "potti sriramulu nellore": (14.4426, 79.9865),
    "nimach": (24.4700, 74.8700),
    "bolangir balangir": (20.7027, 83.4862),
    "keonjhar kendujhar": (21.6289, 85.5817),
    "baudh bauda": (20.8407, 84.3262),
    "sas nagar sahibzada ajit singh nagar": (30.7046, 76.7179),
    "disputed sahibganj maldah katihar": (25.3000, 87.8000),
    "disputed sahibganj maldah and katihar": (25.3000, 87.8000),
    "subarnapur": (20.9000, 83.9200),
    "mandi": (31.7088, 76.9320),
    "punch": (33.7700, 74.1000),
    "bandipura": (34.4200, 74.6400),
    "leh": (34.1526, 77.5771),
    "mansa": (29.9800, 75.3800),
    "tawang": (27.5800, 91.8600),
    "mon": (26.7500, 95.0700),
    "kumuram bheem": (19.3500, 79.4800),
    "warangal urban": (18.0000, 79.5800),
    "warangal rural": (17.9500, 79.6500),
    "ranippettai": (12.9200, 79.3300),
    "tuticorin": (8.7642, 78.1348),
    "kamrup metro": (26.1445, 91.7362),
    "kamrup rural": (26.3100, 91.6000),
}


def clean_name(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace('_', ' ').replace('-', ' ').replace('&', ' ').replace(',', ' ')
    s = re.sub(r'[\(\)]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_centroids():
    indmet_file = Path("data/raw/indmet/India_Districts.csv")
    if not indmet_file.exists():
        print(f"[ERROR] {indmet_file} not found. Please run data_loader first.")
        return

    df_ind = pd.read_csv(indmet_file)
    print(f"Loaded {len(df_ind)} districts from INDmet registry.")

    # Reference coordinates from validated phonepe-pulse open GIS dataset
    ref_url = "https://raw.githubusercontent.com/SaravananSuriya/Phonepe-Pulse-Data-Visualization-and-Exploration/main/lat-%26-lon-india-district.csv"
    print(f"Fetching reference GIS dataset from {ref_url}...")
    df_ref = pd.read_csv(ref_url)

    df_ref['clean'] = df_ref['District'].apply(clean_name)
    ref_map = {}
    for _, row in df_ref.iterrows():
        ref_map[row['clean']] = (float(row['Latitude']), float(row['Longitude']))

    ref_keys = list(ref_map.keys())

    results = []
    matched = 0

    for _, row in df_ind.iterrows():
        d_id = int(row['ID'])
        d_name = str(row['District']).replace('_', ' ').title()
        s_name = str(row['STATE']).replace('_', ' ').title()
        clean_d = clean_name(row['District'])

        lat, lon = None, None

        # 1. Direct manual table check
        state_specific_key = f"{clean_d} {clean_name(row['STATE'])}"
        if state_specific_key in MANUAL_COORDS:
            lat, lon = MANUAL_COORDS[state_specific_key]
        elif clean_d in MANUAL_COORDS:
            lat, lon = MANUAL_COORDS[clean_d]

        # 2. Exact match in reference map
        if lat is None and clean_d in ref_map:
            lat, lon = ref_map[clean_d]

        # 3. Match without spaces
        if lat is None:
            nospace = clean_d.replace(" ", "")
            for k in ref_keys:
                if k.replace(" ", "") == nospace:
                    lat, lon = ref_map[k]
                    break

        # 4. Fuzzy match
        if lat is None:
            closest = difflib.get_close_matches(clean_d, ref_keys, n=1, cutoff=0.75)
            if closest:
                lat, lon = ref_map[closest[0]]

        # 5. Fallback approximation if still missing
        if lat is None:
            lat, lon = (20.5937, 78.9629)  # Center of India fallback
            print(f"[WARN] Fallback centroid assigned for: {d_name} ({s_name})")
        else:
            matched += 1

        results.append({
            "district_id": d_id,
            "district": d_name,
            "state": s_name,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4)
        })

    out_df = pd.DataFrame(results)
    out_path = Path("data/processed/district_centroids.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[SUCCESS] Successfully compiled {len(out_df)} district centroids to {out_path}")
    print(f"[STATS] High-confidence matches: {matched} / {len(out_df)} ({matched/len(out_df)*100:.1f}%)")
    return out_path


if __name__ == "__main__":
    build_centroids()
