import pandas as pd
import geopandas as gpd
import os
import sys
import json
import re

def ward_name_clean(name):
    if pd.isna(name): return ""
    name = str(name).lower()
    # Remove spaces, hyphens, punctuation
    name = re.sub(r'[^a-z0-9]', '', name)
    # Normalize common variations (e.g., 'm corp' removal)
    name = name.replace('mcorp', '').replace('chennai', '')
    return name

def clean_data():
    print("🧹 Phase 2: Cleaning & Standardizing Data...")
    
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Clean Ward Boundaries
    print("  Processing Wards...")
    ward_path = os.path.join(RAW_DIR, "chennai_wards.geojson")
    try:
        wards = gpd.read_file(ward_path)
    except Exception:
        with open(ward_path, 'r') as f:
            data = json.load(f)
        wards = gpd.GeoDataFrame.from_features(data["features"])
    
    # Task 3: Validate full boundary count
    WARD_COUNT_EXPECTED = 200
    if len(wards) < 190: # Allowing slight variation if some small slivers exist, but ~200 is the goal
        print(f"❌ Error: Incomplete administrative boundary file. Found {len(wards)} wards, expected ~{WARD_COUNT_EXPECTED}.")
        sys.exit(1)
        
    # Standardize Column Names
    wards.columns = [c.upper() for c in wards.columns]
    
    if 'GEOMETRY' in wards.columns:
        wards.set_geometry('GEOMETRY', inplace=True)
    elif 'geometry' in wards.columns:
        wards.set_geometry('geometry', inplace=True)

    # Standardize IDs
    if 'WARD_NO' in wards.columns:
        wards['WARD_ID'] = pd.to_numeric(wards['WARD_NO'], errors='coerce').fillna(0).astype(int)
    
    # Clean Names for Task 2
    name_col = next((c for c in wards.columns if 'NAME' in c), None)
    if name_col:
        wards['WARD_NAME_CLEAN'] = wards[name_col].apply(ward_name_clean)
        wards['WARD_NAME'] = wards[name_col]
    
    # Set CRS
    if wards.crs is None:
        wards.set_crs("EPSG:4326", inplace=True)
    wards = wards.to_crs("EPSG:4326")

    # Save Wards with all properties to allow metadata flow
    wards.to_file(os.path.join(PROCESSED_DIR, "wards.geojson"), driver="GeoJSON")

    # 2. Clean Facilities
    print("  Processing Facilities...")
    fac_path = os.path.join(RAW_DIR, "nhm_facilities_chennai.csv")
    facilities = pd.read_csv(fac_path)
    facilities = facilities.drop_duplicates(subset=['latitude', 'longitude'])
    facilities.to_csv(os.path.join(PROCESSED_DIR, "facilities.csv"), index=False)

    # 3. Clean Population
    print("  Processing Population...")
    pop_path = os.path.join(RAW_DIR, "chennai_wards_census_2011.csv")
    pop_df = pd.read_csv(pop_path)
    pop_df.columns = [c.upper() for c in pop_df.columns]
    
    # Sum population by ward ID (using 'WARD NUMBER' in census)
    if 'WARD NUMBER' in pop_df.columns:
        pop_agg = pop_df.groupby('WARD NUMBER')['TOTAL POPULATION'].sum().reset_index()
        pop_agg.rename(columns={'WARD NUMBER': 'WARD_ID', 'TOTAL POPULATION': 'POPULATION'}, inplace=True)
        pop_agg['WARD_ID'] = pop_agg['WARD_ID'].astype(int)
    else:
        # Fallback to name-based join if WARD NUMBER missing
        raise KeyError(f"'WARD NUMBER' not found in Census data.")

    pop_agg.to_csv(os.path.join(PROCESSED_DIR, "population.csv"), index=False)

    print(f"✅ Phase 2 Complete. Files saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    clean_data()
