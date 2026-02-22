import pandas as pd
import geopandas as gpd
import json
import os

def connect_model_to_map():
    print("🔗 Phase 7: Connecting Model to Map...")
    
    GEOJSON_INPUT = "data/processed/chennai_wards_standardized.geojson"
    PRED_PATH = "data/processed/ward_risk_predictions.csv"
    FEAT_PATH = "data/processed/ward_health_features.csv"
    OUTPUT_PATH = "data/chennai/processed/chennai_pilot_intelligence.geojson"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Load Data
    with open(GEOJSON_INPUT, "r") as f:
        geojson_data = json.load(f)
    wards_gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    wards_gdf.set_crs("EPSG:4326", inplace=True)
    
    preds_df = pd.read_csv(PRED_PATH)
    feats_df = pd.read_csv(FEAT_PATH)

    # 2. Merge
    # Standardize ward_id column names for merging
    wards_gdf.columns = [c.lower() for c in wards_gdf.columns]
    preds_df.columns = [c.lower() for c in preds_df.columns]
    feats_df.columns = [c.lower() for c in feats_df.columns]
    
    merged_gdf = wards_gdf.merge(preds_df, on='ward_id', how='left')
    merged_gdf = merged_gdf.merge(feats_df[['ward_id', 'facility_count', 'dist_to_nearest_phc_km', 'population']], on='ward_id', how='left')
    
    # 3. Generate Human Insights
    def generate_insight(row):
        driver = str(row.get('top_risk_driver', 'None')).lower()
        risk_lvl = row.get('risk_category', 'LOW')
        ward = row.get('ward_id', 'Unknown')
        
        if risk_lvl == 'CRITICAL' or risk_lvl == 'HIGH':
            if 'dist' in driver:
                return f"Critical Alert for Ward {ward}: Geographically isolated from PHCs ({row['dist_to_nearest_phc_km']:.1f}km). Immediate mobile medical unit deployment recommended."
            elif 'facility' in driver:
                return f"Critical Alert for Ward {ward}: High density area with only {row['facility_count']} facility/ies. Overcrowding risk at clinics. Priority for auxiliary staff dispatch."
            elif 'cases' in driver or 'seasonal' in driver:
                return f"Outbreak Alert for Ward {ward}: Rising historical case trends detected. Urgent door-to-door larvae survey required."
            else:
                return f"High Risk Alert for Ward {ward}: Population density and environmental factors indicate high outbreak vulnerability."
        else:
            return f"Ward {ward} is currently STABLE. Regular surveillance is sufficient."

    merged_gdf['insight'] = merged_gdf.apply(generate_insight, axis=1)
    
    # Rename risk_category to risk_level for app.py compatibility
    merged_gdf['risk_level'] = merged_gdf['risk_category']
    merged_gdf['timestamp'] = pd.Timestamp.now().isoformat()

    # 4. Save to final GeoJSON
    # We must convert it back to GeoJSON format
    merged_gdf.to_file(OUTPUT_PATH, driver='GeoJSON')
    print(f"✅ Phase 7 Complete: Enriched GeoJSON saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    connect_model_to_map()
