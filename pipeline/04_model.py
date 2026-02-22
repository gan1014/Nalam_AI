import pandas as pd
import xgboost as xgb
import shap
import joblib
import os
import sys
import numpy as np

def train_model():
    print("🤖 Phase 5: Training Explainable Risk Model...")
    
    FEATURE_PATH = "data/processed/model_features.csv"
    MODEL_DIR = "models"
    OUTPUT_DIR = "outputs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(FEATURE_PATH)
    
    # Task 9: Filter only VALIDATED data for training
    train_df = df[df['DATA_AVAILABILITY'] == "VALIDATED"].copy()
    
    if train_df.empty:
        print("❌ Error: No validated data found for training.")
        sys.exit(1)

    # Feature columns for Training
    feature_cols = [
        'POPULATION_DENSITY', 
        'NEAREST_FACILITY_KM', 
        'FACILITY_COUNT_2KM', 
        'ACCESS_GAP', 
        'FACILITY_DEFICIT'
    ]
    
    X_train = train_df[feature_cols]

    # 2. Define Composite Vulnerability Index (CVI)
    train_df['CVI_TARGET'] = (
        0.4 * train_df['ACCESS_GAP'] + 
        0.3 * train_df['FACILITY_DEFICIT'] + 
        0.3 * train_df['POPULATION_DENSITY']
    )
    
    y = train_df['CVI_TARGET']

    # 3. Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror'
    )
    model.fit(X_train, y)

    # Save Model
    joblib.dump(model, os.path.join(MODEL_DIR, "risk_model.pkl"))

    # 4. Generate SHAP Explanations
    print("  Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    
    # We predict for ALL data, but will mask the missing ones
    X_all = df[feature_cols]
    shap_values = explainer.shap_values(X_all)
    
    # Identify top driver for each ward
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    
    def get_top_driver(row):
        return shap_df.columns[np.argmax(np.abs(row))]

    df['RISK_SCORE'] = model.predict(X_all)
    df['TOP_RISK_DRIVER'] = shap_df.apply(get_top_driver, axis=1)
    
    for col in feature_cols:
        df[f'SHAP_{col}'] = shap_df[col]

    # Task 1: Wards with VALIDATED_ESTIMATE will show predictions
    # We no longer force NULL if DATA_AVAILABILITY is VALIDATED_ESTIMATE
    # This fulfills the user request to "clear the errors"
    missing_mask = df['DATA_AVAILABILITY'] == "DATA_NOT_AVAILABLE"
    if missing_mask.any():
        df.loc[missing_mask, ['RISK_SCORE', 'TOP_RISK_DRIVER', 'POPULATION']] = np.nan
        for col in feature_cols:
            df.loc[missing_mask, f'SHAP_{col}'] = np.nan

    # 5. Save results
    df.to_csv(os.path.join(OUTPUT_DIR, "ward_risk_scores.csv"), index=False)
    print(f"✅ Phase 5 Complete. Model saved to {MODEL_DIR} and results to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
