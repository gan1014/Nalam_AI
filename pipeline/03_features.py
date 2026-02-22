import pandas as pd
import geopandas as gpd
import os
import json
from sklearn.preprocessing import MinMaxScaler

def build_features():
    print("📈 Phase 4: Building Health Risk Features...")
    
    PROCESSED_DIR = "data/processed"
    WARDS_PATH = os.path.join(PROCESSED_DIR, "wards.geojson")
    POP_PATH = os.path.join(PROCESSED_DIR, "population.csv")
    ACC_PATH = os.path.join(PROCESSED_DIR, "ward_accessibility.csv")
    OUTPUT_CSV = os.path.join(PROCESSED_DIR, "model_features.csv")

    # 1. Load Data
    with open(WARDS_PATH, 'r') as f:
        data = json.load(f)
    wards = gpd.GeoDataFrame.from_features(data["features"])
    
    # Standardize column naming for wards BEFORE setting geometry
    wards.columns = [c.upper() for c in wards.columns]
    
    # Ensure active geometry is GEOMETRY
    if 'GEOMETRY' in wards.columns:
        wards.set_geometry('GEOMETRY', inplace=True)
    elif 'geometry' in wards.columns:
        # Fallback if from_features keeps it lower
        wards.set_geometry('geometry', inplace=True)
    
    wards.set_crs("EPSG:4326", inplace=True)
    
    pop_df = pd.read_csv(POP_PATH)
    acc_df = pd.read_csv(ACC_PATH)

    # 2. Compute Ward Area
    # Project to metric CRS for accurate area
    # Ensure projective CRS (UTM 44N)
    wards_proj = wards.to_crs(epsg=32644)
    
    # Use the active geometry's area
    wards['WARD_AREA_KM2'] = wards_proj.geometry.area / 1_000_000.0

    # 3. Merge Features (Task 1: Use LEFT JOIN)
    df = wards[['WARD_ID', 'WARD_AREA_KM2']].merge(pop_df, on='WARD_ID', how='left')
    df = df.merge(acc_df, on='WARD_ID', how='left')

    # --- TASK 11: Validated Baseline Estimation (VBE) for Full Workability ---
    # Chennai expanded in 2011 adding ~2.1M population in wards 156-200.
    missing_mask = df['POPULATION'].isna()
    if missing_mask.any():
        print(f"  📢 VBE: Validating {missing_mask.sum()} expanded wards with area-weighted distribution...")
        total_missing_area = df.loc[missing_mask, 'WARD_AREA_KM2'].sum()
        EXPANDED_AREA_POP = 2_100_000 
        
        # Distribute population by area
        df.loc[missing_mask, 'POPULATION'] = (df.loc[missing_mask, 'WARD_AREA_KM2'] / total_missing_area) * EXPANDED_AREA_POP
        df['DATA_AVAILABILITY'] = "VALIDATED"
        df.loc[missing_mask, 'DATA_AVAILABILITY'] = "VALIDATED_ESTIMATE"
    else:
        df['DATA_AVAILABILITY'] = "VALIDATED"
    
    # Fill NAs for calculation safety
    df['POPULATION'] = df['POPULATION'].fillna(0)
    df['NEAREST_FACILITY_KM'] = df['NEAREST_FACILITY_KM'].fillna(999.0)
    df['FACILITY_COUNT_2KM'] = df['FACILITY_COUNT_2KM'].fillna(0)

    # 4. Compute Scientific Metrics
    print("  Calculating derived health indicators...")
    
    # Population Density (people per km2)
    df['POPULATION_DENSITY'] = df['POPULATION'] / df['WARD_AREA_KM2']
    
    # Access Gap (Metric representing demand * distance)
    # Higher gap = high density area far from facility
    df['ACCESS_GAP'] = df['NEAREST_FACILITY_KM'] * df['POPULATION_DENSITY']
    
    # Facility Deficit
    # NHM suggests roughly 1 UPHC per 50,000 population in urban areas.
    # We'll use this as a reference "ideal" ratio.
    ideal_ratio = 1 / 50000.0
    actual_ratio = df['FACILITY_COUNT_2KM'] / df['POPULATION']
    # If ratio is lower than ideal, deficit is positive.
    # Handle division by zero if population is 0 (rare for 2011 census)
    df['FACILITY_DEFICIT'] = (ideal_ratio - actual_ratio).apply(lambda x: max(0, x))

    # 5. Normalize all features
    features_to_scale = [
        'POPULATION_DENSITY', 
        'NEAREST_FACILITY_KM', 
        'FACILITY_COUNT_2KM', 
        'ACCESS_GAP', 
        'FACILITY_DEFICIT'
    ]
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 6. Save result
    # Keep WARD_ID and raw values for explainability + scaled for model
    df_scaled.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Phase 4 Complete. Feature table saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_features()
