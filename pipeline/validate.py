import pandas as pd
import os
import sys

def validate_pipeline():
    print("🔍 Phase 6: Validating Pipeline Results...")
    
    SCORES_PATH = "outputs/ward_risk_scores.csv"
    FEATURES_PATH = "data/processed/model_features.csv"
    
    if not os.path.exists(SCORES_PATH) or not os.path.exists(FEATURES_PATH):
        print("❌ Error: Output files missing.")
        sys.exit(1)
        
    df = pd.read_csv(SCORES_PATH)
    
    # Check 1: No constant columns
    for col in ['POPULATION_DENSITY', 'NEAREST_FACILITY_KM', 'RISK_SCORE']:
        if df[col].std() == 0:
            print(f"❌ Validation Error: Column '{col}' is constant (standard deviation is 0).")
            sys.exit(1)
            
    # Check 2: Distances are not all zero
    if (df['NEAREST_FACILITY_KM'] == 0).all():
        print("❌ Validation Error: All distances are zero. Spatial join likely failed.")
        sys.exit(1)
        
    # Check 3: Risk scores vary
    if df['RISK_SCORE'].nunique() < 2:
        print("❌ Validation Error: Risk scores do not vary across wards.")
        sys.exit(1)

    # Check 4: Row linkage
    if len(df) < 155: # Chennai had at least 155 wards in 2011 census
        print(f"❌ Validation Error: Insufficient wards detected ({len(df)}). Missing data rows.")
        sys.exit(1)

    # Check 5: Sensitivity (Informational)
    # We verify that RISK_SCORE is correlated with expected drivers
    corr = df[['RISK_SCORE', 'ACCESS_GAP']].corr().iloc[0,1]
    if corr < 0.5:
        print(f"⚠️ Warning: Risk score correlation with Access Gap is low ({corr:.2f}). Check model logic.")

    print("✅ Pipeline Validation PASSED.")
    print(f"   Processed {len(df)} wards.")
    print(f"   Risk Score Range: {df['RISK_SCORE'].min():.4f} to {df['RISK_SCORE'].max():.4f}")

if __name__ == "__main__":
    validate_pipeline()
