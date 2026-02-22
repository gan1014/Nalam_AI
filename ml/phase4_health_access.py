import pandas as pd
import geopandas as gpd
import os
import json

def calculate_health_access():
    print("📏 Phase 4: Calculating Geodesic Distances...")
    
    FACILITIES_PATH = "data/raw/nhm_facilities_chennai.csv"
    GEOJSON_PATH = "data/processed/chennai_wards_standardized.geojson"
    OUTPUT_CSV = "data/processed/health_access.csv"

    # 1. Load Wards
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)
    wards_gdf = gpd.GeoDataFrame.from_features(data["features"])
    wards_gdf.set_crs("EPSG:4326", inplace=True)
    
    # 2. Load Facilities
    health_df = pd.read_csv(FACILITIES_PATH)
    facilities_gdf = gpd.GeoDataFrame(
        health_df,
        geometry=gpd.points_from_xy(health_df.longitude, health_df.latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=32644)
    if 'ward_id' in facilities_gdf.columns:
        facilities_gdf = facilities_gdf.drop(columns=['ward_id'])

    # 3. Project Wards and get centroids
    wards_proj = wards_gdf.to_crs(epsg=32644)
    wards_proj['centroid_geom'] = wards_proj.geometry.centroid
    
    # Standardize ward_id column name
    wards_proj.columns = [c.lower() for c in wards_proj.columns]
    
    # 4. Nearest Join
    centroids_gdf = wards_proj[['ward_id', 'centroid_geom']].rename(columns={'centroid_geom': 'geometry'}).set_geometry('geometry')
    
    nearest = gpd.sjoin_nearest(centroids_gdf, facilities_gdf, distance_col="dist_m", how="left")
    
    # 5. Result
    access_df = nearest.groupby('ward_id')['dist_m'].min().reset_index()
    access_df['dist_to_nearest_phc_km'] = access_df['dist_m'] / 1000.0

    access_df[['ward_id', 'dist_to_nearest_phc_km']].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    calculate_health_access()
