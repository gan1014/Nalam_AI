import pandas as pd
import geopandas as gpd
import json
import os

def standardize_ward_data():
    print("🧹 Standardizing Ward Data...")
    
    CENSUS_PATH = "data/raw/chennai_wards_census_2011.csv"
    GEOJSON_PATH = "data/raw/chennai_wards.geojson"
    OUTPUT_FOLDER = "data/processed/"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Process Census Data
    # The census data is at Enumeration Block level, we need to sum by Ward Number
    census_df = pd.read_csv(CENSUS_PATH)
    census_ward = census_df.groupby('Ward Number')['Total Population'].sum().reset_index()
    census_ward.rename(columns={'Ward Number': 'ward_id', 'Total Population': 'population'}, inplace=True)
    
    # 2. Process GeoJSON
    try:
        wards_gdf = gpd.read_file(GEOJSON_PATH)
    except Exception:
        with open(GEOJSON_PATH, 'r') as f:
            data = json.load(f)
        wards_gdf = gpd.GeoDataFrame.from_features(data["features"])
    
    # Standardize Ward ID in GeoJSON
    if 'Ward_No' in wards_gdf.columns:
        wards_gdf['ward_id'] = wards_gdf['Ward_No'].astype(int)
    elif 'WARD_NO' in wards_gdf.columns:
        wards_gdf['ward_id'] = wards_gdf['WARD_NO'].astype(int)
    
    # 3. Join Census to GeoJSON to verify consistency
    final_wards = wards_gdf.merge(census_ward, on='ward_id', how='left')
    
    # Fill missing population with average if necessary (though we aim for real data)
    avg_pop = final_wards['population'].mean()
    final_wards['population'] = final_wards['population'].fillna(avg_pop)

    # Save standardized ward features (initial)
    final_wards[['ward_id', 'population']].to_csv(os.path.join(OUTPUT_FOLDER, "ward_population.csv"), index=False)
    
    # Save cleaned GeoJSON
    final_wards.to_file(os.path.join(OUTPUT_FOLDER, "chennai_wards_standardized.geojson"), driver='GeoJSON')
    
    print(f"✅ Phase 2 Complete: Standardized {len(final_wards)} wards.")

if __name__ == "__main__":
    standardize_ward_data()
