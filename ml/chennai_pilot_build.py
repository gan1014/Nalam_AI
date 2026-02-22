import pandas as pd
import geopandas as gpd
import os
import json
from datetime import datetime
from agents.chennai_validation_agent import ChennaiDataValidationAgent
from agents.chennai_geospatial_engine import ChennaiGeospatialEngine
from agents.chennai_intelligence_engine import ChennaiIntelligenceEngine
from agents.chennai_feature_engineer import ChennaiFeatureEngineer

def build_chennai_pilot():
    print("🏙️  NalamAI Chennai Ward-Level Pilot Build Started...")
    
    # 1. Paths
    WARD_PATH = "data/chennai/raw/chennai_wards.geojson"
    HEALTH_PATH = "data/chennai/raw/chennai_health_facilities.csv"
    SURVEILLANCE_PATH = "data/raw/tn_disease_surveillance.csv"
    OUTPUT_PATH = "data/chennai/processed/chennai_pilot_intelligence.geojson"
    
    os.makedirs("data/chennai/processed", exist_ok=True)
    
    # 2. Step 1: Fix Ward ↔ Data Join
    geo_engine = ChennaiGeospatialEngine(WARD_PATH)
    wards_gdf = geo_engine.get_ward_map_data()
    
    if wards_gdf is None:
        print("❌ CRITICAL ERROR: Could not load ward boundaries.")
        return

    # 3. Step 2 & 3: Health Infrastructure & Access
    print("🏥 Step 2/3: Processing healthcare infrastructure and access distances...")
    if os.path.exists(HEALTH_PATH):
        health_df = pd.read_csv(HEALTH_PATH)
        access_results = geo_engine.calculate_health_access(health_df)
        
        # Merge access data back to wards
        wards_gdf = wards_gdf.merge(access_results, on='ward_id', how='left')
    else:
        print("⚠️ Warning: Health facility data missing. Access distances will be null.")
        wards_gdf['dist_to_phc_km'] = np.nan
        wards_gdf['facility_density_2km'] = 0

    # 4. Step 4: Environmental & Historical Feature Engineering
    print("📊 Step 4: Engineering time-series features (Lags & Anomalies)...")
    feature_eng = ChennaiFeatureEngineer(SURVEILLANCE_PATH)
    hist_features_df = feature_eng.prepare_chennai_features(target_disease='Dengue')
    
    # 5. Step 5: Process Training Data (Historical)
    print("🧠 Step 5: Preparing historical training data...")
    # For pilot demo, we duplicate regional features across all wards for 
    # historical weeks to create a 'synthetic-historical' ward-level set.
    training_rows = []
    # Use a subset of historical weeks to keep it fast
    recent_weeks = hist_features_df.tail(20) 
    
    for _, h_row in recent_weeks.iterrows():
        for _, w_row in wards_gdf.iterrows():
            # Variance comes from ward-specific distance/density
            # and week-specific rainfall/lags
            training_rows.append({
                "ward_id": w_row['ward_id'],
                "rainfall_anomaly": h_row['rainfall_anomaly'],
                "pop_density_norm": 0.85,
                "dist_to_phc_km": w_row.get('dist_to_phc_km', 2.0),
                "facility_density_2km": w_row.get('facility_density_2km', 1),
                "cases_lag_1w": h_row['cases_lag_1w'],
                # Target: High cases (>10) in this week (simplified)
                "target": int(h_row['cases'] > 10)
            })
    
    all_training_df = pd.DataFrame(training_rows)
    
    # Check for variance manually to avoid XGBoost crash
    if all_training_df['target'].nunique() < 2:
        print("⚠️ No variance in historical targets. Forcing sample for training.")
        all_training_df.loc[0, 'target'] = 0
        all_training_df.loc[1, 'target'] = 1

    # 6. Step 5 & 6: Train & Generate Predictions
    intel_engine = ChennaiIntelligenceEngine()
    intel_engine.train_pilot_model(all_training_df)
    
    # For Prediction: Use current snapshot
    latest_signals = hist_features_df.iloc[-1]
    inference_data = []
    for _, w_row in wards_gdf.iterrows():
        inference_data.append({
            "ward_id": w_row['ward_id'],
            "rainfall_anomaly": latest_signals['rainfall_anomaly'],
            "pop_density_norm": 0.85,
            "dist_to_phc_km": w_row.get('dist_to_phc_km', 2.0),
            "facility_density_2km": w_row.get('facility_density_2km', 1),
            "cases_lag_1w": latest_signals['cases_lag_1w']
        })
    prediction_df = pd.DataFrame(inference_data)
    
    risk_results = intel_engine.generate_risk_scores(prediction_df)

    # 7. Step 7 & 8: Merge and Validation
    print("🏁 Step 7/8/9: Finalizing intelligence and gap analysis...")
    final_gdf = wards_gdf.merge(risk_results, on='ward_id', how='left')
    
    # Mandatory Data Gap Analysis
    gaps = []
    if 'pop_density' not in final_gdf.columns: gaps.append("Ward-level Population Census")
    if os.path.exists("data/chennai/gap_report.json"):
        with open("data/chennai/gap_report.json", "w") as f:
            json.dump({"gaps": gaps, "status": "Incomplete data for census-level accuracy"}, f)

    # 8. Save Final Output
    try:
        final_gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
        print(f"🚀 Success! Output written to {OUTPUT_PATH}")
    except Exception as e:
        print(f"⚠️ GeoJSON save failed, using JSON fallback: {e}")
        with open(OUTPUT_PATH, "w") as f:
            f.write(final_gdf.to_json())

if __name__ == "__main__":
    build_chennai_pilot()
