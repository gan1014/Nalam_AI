import pandas as pd
import json
import os

def audit():
    print("--- ROBUST DATA AUDIT ---")
    ward_path = "data/raw/chennai_wards.geojson"
    census_path = "data/raw/chennai_wards_census_2011.csv"
    
    # 1. Ward Boundaries (using json to be safe from gpd issues)
    if os.path.exists(ward_path):
        with open(ward_path, 'r') as f:
            data = json.load(f)
        features = data.get("features", [])
        print(f"Total Wards in GeoJSON: {len(features)}")
        if features:
            print(f"GeoJSON Properties: {list(features[0]['properties'].keys())}")
    
    # 2. Census Data
    if os.path.exists(census_path):
        census = pd.read_csv(census_path)
        print(f"Total Entries in Census: {len(census)}")
        print(f"Census Columns: {list(census.columns)}")
        # Check for ward names
        name_col = [c for c in census.columns if 'name' in c.lower()]
        if name_col:
            print(f"Sample Names: {census[name_col[0]].head().tolist()}")
    
if __name__ == "__main__":
    audit()
