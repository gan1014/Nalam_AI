
import geopandas as gpd
import os

WARD_PATH = "data/chennai/raw/chennai_wards.geojson"
print(f"Checking {WARD_PATH}...")
if os.path.exists(WARD_PATH):
    try:
        gdf = gpd.read_file(WARD_PATH)
        print(f"Success! Loaded {len(gdf)} rows.")
        print(gdf.columns)
    except Exception as e:
        print(f"Failed! Error: {e}")
else:
    print("File not found.")
