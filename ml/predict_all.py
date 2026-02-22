
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import sys
import traceback

# Paths
PROCESSED_DATA_PATH = 'nalamai/data/processed/train_scaled.csv'
MODEL_PATH = 'nalamai/models/xgb_risk.pkl'
SCALER_PATH = 'nalamai/models/scaler.pkl'
LE_DISTRICT_PATH = 'nalamai/models/le_district.pkl'
LE_DISEASE_PATH = 'nalamai/models/le_disease.pkl'
LATEST_PREDICTIONS_PATH = 'nalamai/data/processed/latest_predictions.csv'

# Add parent dir to path for backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nalamai.backend import db

def main():
    print("Running Prediction Engine...")
    
    try:
        # Load Artifacts
        xgb = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le_district = joblib.load(LE_DISTRICT_PATH)
        le_disease = joblib.load(LE_DISEASE_PATH)
        
        # Load latest data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        last_date = df['date'].max()
        latest_data = df[df['date'] == last_date].copy()
        
        print(f"Generating predictions for {len(latest_data)} district-disease pairs...")
        print(f"Columns in CSV: {latest_data.columns.tolist()}")
        if hasattr(xgb, 'feature_names_in_'):
            print(f"Model expects: {xgb.feature_names_in_.tolist()}")
        
        # Prepare Features
        # EXCLUDE target cols but INCLUDE encoded cols if model needs them (it does!)
        # Check what model expects
        if hasattr(xgb, 'feature_names_in_'):
            feature_cols = xgb.feature_names_in_.tolist()
            # Verify they exist
            missing = [c for c in feature_cols if c not in latest_data.columns]
            if missing:
                print(f"❌ Missing features in data: {missing}")
                return
        else:
             # Fallback
             feature_cols = [col for col in latest_data.columns if col not in ['date', 'district', 'disease', 'cases', 'risk_label']]
        
        print(f"Using features ({len(feature_cols)}): {feature_cols}")

        # Predict Risk Probability
        X = latest_data[feature_cols]
        print(f"X shape: {X.shape}")
        
        risk_probs = xgb.predict_proba(X)[:, 1]
        
        # Prepare Output
        results = []
        
        # Extract the original string values directly from the DataFrame! 
        # The DataFrame contains the actual unencoded 'district' and 'disease' columns.
        districts = latest_data['district'].values
        diseases = latest_data['disease'].values
        
        for i, (idx, row) in enumerate(latest_data.iterrows()):
            prob = risk_probs[i]
            
            if prob > 0.7:
                risk_level = "HIGH"
                db.log_alert(districts[i], diseases[i], "HIGH", "Logged")
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            predicted_cases = int(row['cases'] * (1 + prob))
            
            results.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'district': districts[i],
                'disease': diseases[i],
                'risk_score': round(prob, 4),
                'risk_level': risk_level,
                'predicted_cases': predicted_cases
            })
            
            db.insert_prediction(
                datetime.now().strftime('%Y-%m-%d'),
                districts[i],
                diseases[i],
                prob,
                risk_level,
                predicted_cases
            )

        # Save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(LATEST_PREDICTIONS_PATH, index=False)
        
        print(f"✅ Saved predictions to {LATEST_PREDICTIONS_PATH}")
        print("Risk Summary:")
        if not results_df.empty:
            print(results_df['risk_level'].value_counts())
            
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
