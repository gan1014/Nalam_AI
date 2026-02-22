import pandas as pd
import geopandas as gpd
import os

def audit():
    print("--- DATA AUDIT ---")
    ward_path = "data/raw/chennai_wards.geojson"
    census_path = "data/raw/chennai_wards_census_2011.csv"
    
    # 1. Ward Boundaries
    wards = gpd.read_file(ward_path)
    print(f"Total Wards in GeoJSON: {len(wards)}")
    print(f"GeoJSON Columns: {list(wards.columns)}")
    
    # 2. Census Data
    census = pd.read_csv(census_path)
    print(f"Total Entries in Census: {len(census)}")
    print(f"Census Columns: {list(census.columns)}")
    
    # 3. Accessibility Data (already processed in previous turns?)
    # Let's check if ward_accessibility.csv exists from previous runs
    acc_path = "data/processed/ward_accessibility.csv"
    if os.path.exists(acc_path):
        acc = pd.read_csv(acc_path)
        print(f"Total Wards with Accessibility: {len(acc)}")
    
if __name__ == "__main__":
    audit()
