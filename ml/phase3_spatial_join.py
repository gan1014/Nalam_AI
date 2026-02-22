import pandas as pd
import geopandas as gpd
import os

def map_facilities_to_wards():
    print("🗺️  Performing Spatial Join: Facilities -> Wards...")
    
    FACILITIES_PATH = "data/raw/nhm_facilities_chennai.csv"
    GEOJSON_PATH = "data/processed/chennai_wards_standardized.geojson"
    OUTPUT_CSV = "data/processed/facility_count_by_ward.csv"

    # 1. Load Facilities
    health_df = pd.read_csv(FACILITIES_PATH)
    facilities_gdf = gpd.GeoDataFrame(
        health_df,
        geometry=gpd.points_from_xy(health_df.longitude, health_df.latitude),
        crs="EPSG:4326"
    )

    # 2. Load Standardized Wards
    try:
        wards_gdf = gpd.read_file(GEOJSON_PATH)
    except Exception:
        import json
        with open(GEOJSON_PATH, "r") as f:
            data = json.load(f)
        wards_gdf = gpd.GeoDataFrame.from_features(data["features"])
        wards_gdf.set_crs("EPSG:4326", inplace=True)
    
    if wards_gdf.crs is None:
        wards_gdf.set_crs("EPSG:4326", inplace=True)

    # 3. Spatial Join
    # Drop existing ward_id in facilities if present to avoid collision, 
    # but we want to VERIFY if the point actually falls in the assigned ward.
    # For official requirement, we must spatially assign.
    if 'ward_id' in facilities_gdf.columns:
        facilities_gdf = facilities_gdf.drop(columns=['ward_id'])
    
    joined_gdf = gpd.sjoin(facilities_gdf, wards_gdf[['ward_id', 'geometry']], how="inner", predicate="within")

    # 4. Count by Ward
    facility_counts = joined_gdf.groupby('ward_id').size().reset_index(name='facility_count')
    
    # 5. Ensure all wards are present (even with 0 facilities)
    all_wards = wards_gdf[['ward_id']].copy()
    final_counts = all_wards.merge(facility_counts, on='ward_id', how='left').fillna(0)
    final_counts['facility_count'] = final_counts['facility_count'].astype(int)

    # 6. Save Output
    final_counts.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Phase 3 Complete: Mapped facilities to {len(final_counts)} wards. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    map_facilities_to_wards()
