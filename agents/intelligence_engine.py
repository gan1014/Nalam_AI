
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import sys
import json

from agents.risk_agent import RiskIntelligenceAgent
from agents.recommendation_agent import RecommendationEngine
from agents.data_agent import DataTrustAgent

# Paths
PROCESSED_DATA_PATH = 'data/processed/train_scaled.csv'
MODEL_PATH = 'models/xgb_risk.pkl'
LATEST_PREDICTIONS_PATH = 'data/processed/latest_predictions.csv'

class IntelligenceEngine:
    def __init__(self):
        self.risk_agent = RiskIntelligenceAgent()
        self.rec_engine = RecommendationEngine()
        self.data_agent = DataTrustAgent()
        
    def run(self):
        print("🚀 NalamAI Autonomous Intelligence Engine Starting...")
        
        # 1. Verify Data Source (Agent 1/2)
        self.data_agent.verify_source("TN_HEALTH_DHS")
        
        # 2. Load Data
        df = pd.read_csv(PROCESSED_DATA_PATH)
        self.data_agent.validate_dataset(df, "TN_HEALTH_DHS")
        
        # 3. Load Model
        xgb = joblib.load(MODEL_PATH)
        
        # 4. Filter for latest date
        last_date = df['date'].max()
        latest_data = df[df['date'] == last_date].copy()
        
        # 5. Feature Engineering (Simplified Agent 3/4)
        if hasattr(xgb, 'feature_names_in_'):
            feature_cols = xgb.feature_names_in_.tolist()
        else:
            feature_cols = [col for col in latest_data.columns if col not in ['date', 'district', 'disease', 'cases', 'risk_label']]
        
        # 6. Generate Risk Scores
        X = latest_data[feature_cols]
        risk_probs = xgb.predict_proba(X)[:, 1]
        
        # 7. Categorize and Recommend (Agent 5 & 6)
        results = []
        districts = latest_data['district'].values
        diseases = latest_data['disease'].values
        
        for i, prob in enumerate(risk_probs):
            risk_level = self.risk_agent.categorize_risk(prob)
            recs = self.rec_engine.get_recommendations(risk_level, diseases[i])
            
            results.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'district': districts[i],
                'disease': diseases[i],
                'risk_score': round(prob, 4),
                'risk_level': risk_level,
                'predicted_cases': int(latest_data.iloc[i]['cases'] * (1 + prob)),
                'recommendations': " | ".join(recs)
            })
            
        # 8. Save Intelligent Output
        results_df = pd.DataFrame(results)
        results_df.to_csv(LATEST_PREDICTIONS_PATH, index=False)
        
        # 9. Log Lineage (Governance)
        self.data_agent.log_lineage("INTELLIGENCE_GENERATION", {
            "model": "xgb_risk.pkl",
            "version": "v1.0.TN",
            "records_processed": len(results),
            "output": LATEST_PREDICTIONS_PATH
        })
        
        print(f"✅ Intelligence Summary Generated: {len(results)} locations processed.")
        print(results_df['risk_level'].value_counts())

if __name__ == "__main__":
    engine = IntelligenceEngine()
    engine.run()
