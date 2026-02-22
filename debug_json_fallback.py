
import geopandas as gpd
import json
import os
import pandas as pd

path = 'data/chennai/processed/chennai_pilot_intelligence.geojson'
try:
    with open(path, 'r') as f:
        c_data = json.load(f)
    print(f"JSON loaded. Top level keys: {list(c_data.keys())}")
    
    c_gdf = gpd.GeoDataFrame.from_features(c_data['features'])
    print(f"GDF created from features. Rows: {len(c_gdf)}")
    print(f"Columns: {list(c_gdf.columns)}")
    
    if not c_gdf.empty:
        c_gdf.set_crs("EPSG:4326", inplace=True)
        print("CRS set.")
    
    has_chennai = not c_gdf.empty
    print(f"has_chennai: {has_chennai}")
    
except Exception as e:
    print(f"Error: {e}")
