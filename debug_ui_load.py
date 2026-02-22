
import geopandas as gpd
import json
import os

path = 'data/chennai/processed/chennai_pilot_intelligence.geojson'
print(f"Checking path: {os.path.abspath(path)}")
print(f"Exists: {os.path.exists(path)}")

try:
    try:
        c_gdf = gpd.read_file(path)
        print(f"Loaded via gpd.read_file. Rows: {len(c_gdf)}")
    except Exception as e:
        print(f"gpd.read_file failed: {e}")
        # Robust JSON fallback
        with open(path, 'r') as f:
            c_data = json.load(f)
        c_gdf = gpd.GeoDataFrame.from_features(c_data['features'])
        c_gdf.set_crs("EPSG:4326", inplace=True)
        print(f"Loaded via JSON fallback. Rows: {len(c_gdf)}")
        
    has_chennai = not c_gdf.empty
    print(f"has_chennai: {has_chennai}")
except Exception as e:
    print(f"Everything failed: {e}")
