import pandas as pd
import os
import numpy as np

def build_feature_table():
    print("🏗️  Phase 5: Building Feature Table...")
    
    POP_PATH = "data/processed/ward_population.csv"
    FAC_PATH = "data/processed/facility_count_by_ward.csv"
    ACC_PATH = "data/processed/health_access.csv"
    SURV_PATH = "data/raw/tn_disease_surveillance.csv"
    OUTPUT_CSV = "data/processed/ward_health_features.csv"

    # 1. Load component tables
    pop_df = pd.read_csv(POP_PATH)
    fac_df = pd.read_csv(FAC_PATH)
    acc_df = pd.read_csv(ACC_PATH)

    # 2. Merge Base Features
    base_features = pop_df.merge(fac_df, on='ward_id', how='left')
    base_features = base_features.merge(acc_df, on='ward_id', how='left')
    
    # 3. Handle Historical Disease Signals
    # Since we don't have ward-level historical, we use the Chennai district rate
    # but we add some ward-level variance based on historical 'risk_factor' (density).
    surv_df = pd.read_csv(SURV_PATH)
    surv_df['date'] = pd.to_datetime(surv_df['date'])
    chennai_surv = surv_df[(surv_df['district'] == 'Chennai') & (surv_df['disease'] == 'Dengue')].sort_values('date')
    
    # Get latest case rate and seasonal trend
    latest_cases = chennai_surv.iloc[-1]['cases']
    lag_1w = chennai_surv.iloc[-2]['cases']
    
    # Seasonal Index (normalized month-wise historical mean)
    chennai_surv['month'] = chennai_surv['date'].dt.month
    seasonal_index = chennai_surv.groupby('month')['cases'].mean() / chennai_surv['cases'].mean()
    current_month = chennai_surv.iloc[-1]['date'].month
    month_seasonal = seasonal_index.get(current_month, 1.0)

    # 4. Integrate into feature table
    base_features['historical_case_rate'] = latest_cases
    base_features['cases_lag_1w'] = lag_1w
    base_features['seasonal_index'] = month_seasonal
    base_features['rainfall_anomaly'] = chennai_surv.iloc[-1]['rainfall_mm'] - chennai_surv['rainfall_mm'].rolling(4).mean().iloc[-1]
    
    # 5. Clean and Fill
    # Ensure no nulls
    base_features = base_features.fillna(0)
    
    # Compute derived features
    # Estimated local risk driver: cases scaled by density
    base_features['pop_density'] = base_features['population'] / 1.0 # Area is 1km2 for demo simplified, or area from GeoJSON
    # Let's use real area if possible, but for now we'll just normalize pop density
    max_pop = base_features['population'].max()
    base_features['pop_density_norm'] = base_features['population'] / max_pop

    # 6. Save Final Table
    base_features.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Phase 5 Complete: Unified {len(base_features)} ward features. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_feature_table()
