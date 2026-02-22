
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# Paths
PROCESSED_DATA_PATH = 'nalamai/data/processed/train_scaled.csv'
MODEL_PATH = 'nalamai/models/xgb_risk.pkl'
SHAP_PLOTS_DIR = 'nalamai/frontend/shap_plots'

def main():
    print("Generating SHAP Explanations...")
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        xgb = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Data or model not found. Run train_xgb.py first.")
        return

    # Prepare data for SHAP
    X = df.drop(['risk_label', 'cases', 'date', 'district', 'disease'], axis=1)
    
    # Sample for speed (SHAP is slow)
    X_sample = X.sample(n=500, random_state=42)
    
    # Explain
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)
    
    # Global Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f'{SHAP_PLOTS_DIR}/global_importance.png', bbox_inches='tight')
    plt.close()
    print(f"Saved global_importance.png")
    
    # Feature Importance Bar Chart
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(f'{SHAP_PLOTS_DIR}/feature_bar.png', bbox_inches='tight')
    plt.close()
    print(f"Saved feature_bar.png")
    
    # Per-District Analysis (Top 5 Districts)
    top_districts = ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem']
    le_district = joblib.load('nalamai/models/le_district.pkl')
    
    for dist in top_districts:
        try:
            dist_code = le_district.transform([dist])[0]
            district_indices = df[df['district_encoded'] == dist_code].index
            
            if len(district_indices) > 0:
                # Get sample indices that overlap with our X_sample
                # (OR just re-calculate SHAP for specific district samples)
                X_dist = X.loc[district_indices].sample(n=min(50, len(district_indices)), random_state=42)
                shap_values_dist = explainer.shap_values(X_dist)
                
                plt.figure()
                shap.summary_plot(shap_values_dist, X_dist, plot_type="bar", show=False)
                plt.title(f"Feature Importance for {dist}")
                plt.savefig(f'{SHAP_PLOTS_DIR}/{dist.lower()}_importance.png', bbox_inches='tight')
                plt.close()
                print(f"Saved {dist} importance plot")
        except Exception as e:
            print(f"Skipping {dist}: {e}")

    print("✅ SHAP Explanation Generation Complete.")

if __name__ == "__main__":
    main()
