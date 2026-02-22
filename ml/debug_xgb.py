
import joblib
import pandas as pd
import xgboost as xgb
import os

MODEL_PATH = 'nalamai/models/xgb_risk.pkl'
DATA_PATH = 'nalamai/data/processed/train_scaled.csv'

def main():
    print("Loading model...")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {type(model)}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names in model ({len(model.feature_names_in_)}):")
            print(model.feature_names_in_)
        else:
            print("Model has no feature_names_in_ attribute.")
            
        print("\nLoading data...")
        df = pd.read_csv(DATA_PATH)
        print(f"Data columns ({len(df.columns)}):")
        print(df.columns.tolist())
        
        # Check for intersection
        if hasattr(model, 'feature_names_in_'):
            missing = [c for c in model.feature_names_in_ if c not in df.columns]
            print(f"\nMissing columns in data: {missing}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
