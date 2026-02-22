import pandas as pd
import geopandas as gpd
import os
import traceback

def debug_phase3():
    try:
        FACILITIES_PATH = "data/raw/nhm_facilities_chennai.csv"
        GEOJSON_PATH = "data/processed/chennai_wards_standardized.geojson"
        
        print(f"Reading {FACILITIES_PATH}")
        df = pd.read_csv(FACILITIES_PATH)
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        print(f"Reading {GEOJSON_PATH}")
        gdf_wards = gpd.read_file(GEOJSON_PATH)
        
        print("Performing sjoin...")
        joined = gpd.sjoin(gdf_points, gdf_wards, how="inner", predicate="within")
        print(f"Joined {len(joined)} points.")
        
    except Exception as e:
        print("❌ Error caught:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_phase3()
