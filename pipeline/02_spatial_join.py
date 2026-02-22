import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import json

def spatial_join():
    print("📏 Phase 3: Performing Real Spatial Join...")
    
    PROCESSED_DIR = "data/processed"
    WARDS_PATH = os.path.join(PROCESSED_DIR, "wards.geojson")
    FAC_PATH = os.path.join(PROCESSED_DIR, "facilities.csv")
    OUTPUT_CSV = os.path.join(PROCESSED_DIR, "ward_accessibility.csv")

    # 1. Load Data
    with open(WARDS_PATH, 'r') as f:
        data = json.load(f)
    
    # from_features creates a DataFrame
    df_wards = pd.DataFrame([f['properties'] for f in data['features']])
    geoms = [Point(f['geometry']['coordinates']) if f['geometry']['type'] == 'Point' else None for f in data['features']]
    # Wait, Wards are POLYGONS.
    
    # Better to use GeoDataFrame.from_features directly
    wards = gpd.GeoDataFrame.from_features(data["features"])
    
    # EXPLICITLY set geometry
    if 'geometry' in wards.columns:
        wards.set_geometry('geometry', inplace=True)
    
    wards.set_crs("EPSG:4326", inplace=True)
    
    # Standardize column naming for wards
    wards.columns = [c.upper() for c in wards.columns]
    if 'GEOMETRY' in wards.columns:
        wards.set_geometry('GEOMETRY', inplace=True)

    # 2. Load Facilities
    fac_df = pd.read_csv(FAC_PATH)
    facilities = gpd.GeoDataFrame(
        fac_df, 
        geometry=gpd.points_from_xy(fac_df.longitude, fac_df.latitude),
        crs="EPSG:4326"
    )

    # 3. Project to UTM (Chennai is UTM 44N)
    wards_proj = wards.to_crs(epsg=32644)
    fac_proj = facilities.to_crs(epsg=32644)

    # 4. Compute Centroids
    # Use the projected geometry to get the centroid in meters
    wards_proj['CENTROID_GEOM'] = wards_proj.geometry.centroid
    
    # Create a new GDF for centroids to do sjoin_nearest
    centroids_gdf = wards_proj[['WARD_ID', 'CENTROID_GEOM']].copy()
    centroids_gdf.set_geometry('CENTROID_GEOM', inplace=True)

    # 5. Nearest Facility Distance
    print("  Calculating nearest facility distances...")
    # sjoin_nearest returns the distance in the CRS units (meters)
    nearest = gpd.sjoin_nearest(centroids_gdf, fac_proj, distance_col="DIST_M", how="left")
    
    # Aggregate to ensure one row per ward
    access_df = nearest.groupby('WARD_ID')['DIST_M'].min().reset_index()
    access_df['NEAREST_FACILITY_KM'] = access_df['DIST_M'] / 1000.0

    # 6. Facility Count within 2km (Density)
    print("  Calculating facility density (2km buffer)...")
    # Buffer each ward'S ACTUAL GEOMETRY or centroid? 
    # Requirement says "Count facilities within 2km buffer" of ward (usually the polygon or centroid).
    # Let's use the ward polygon for "facilities inside or near"
    
    ward_buffers = wards_proj.copy()
    ward_buffers['buffered_geom'] = ward_buffers.geometry.buffer(2000) # 2km extension
    ward_buffers.set_geometry('buffered_geom', inplace=True)
    
    density_join = gpd.sjoin(fac_proj, ward_buffers[['WARD_ID', 'buffered_geom']], how="inner", predicate="within")
    density_counts = density_join.groupby('WARD_ID').size().reset_index(name='FACILITY_COUNT_2KM')
    
    # Task 1: Initialize result from FULL ward list to ensure no dropout
    full_ward_ids = wards_proj[['WARD_ID']].drop_duplicates()
    final_access = full_ward_ids.merge(access_df, on='WARD_ID', how='left')
    final_access = final_access.merge(density_counts, on='WARD_ID', how='left')
    
    # Handle missing data (Task 1 handling)
    final_access['NEAREST_FACILITY_KM'] = final_access['NEAREST_FACILITY_KM'].fillna(999.0) # Using 999 as placeholder for "Very Far"
    final_access['FACILITY_COUNT_2KM'] = final_access['FACILITY_COUNT_2KM'].fillna(0).astype(int)

    # 7. Save Result
    final_access[['WARD_ID', 'NEAREST_FACILITY_KM', 'FACILITY_COUNT_2KM']].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Phase 3 Complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    spatial_join()
