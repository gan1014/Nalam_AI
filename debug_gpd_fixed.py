
import json
import geopandas as gpd
import pandas as pd

WARD_PATH = "data/chennai/raw/chennai_wards.geojson"
print(f"Checking {WARD_PATH} via JSON fallback...")

try:
    with open(WARD_PATH, "r") as f:
        data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    gdf.set_crs("EPSG:4326", inplace=True)
    print(f"Success! Loaded {len(gdf)} rows via JSON.")
    print(gdf.columns)
    
    # Test project
    gdf_proj = gdf.to_crs(epsg=32644)
    print("Projected successfully.")
except Exception as e:
    print(f"Failed! Error: {e}")
