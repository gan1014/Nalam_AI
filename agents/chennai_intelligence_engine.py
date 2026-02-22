
import pandas as pd
import numpy as np
import os
import joblib
import shap
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class ChennaiIntelligenceEngine:
    """
    MODULE D — ML RISK MODEL
    MODULE F — DECISION SUPPORT PANEL
    """
    
    def __init__(self, model_path="models/chennai/ward_risk_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def generate_risk_scores(self, ward_features_df):
        """
        Input features: rainfall_anomaly, pop_density, dist_to_phc, facility_density
        Output: Risk Score per Ward: LOW | MODERATE | HIGH | CRITICAL
        """
        print("🧠 Running Chennai Ward-Level Risk Intelligence (XGBoost + SHAP)...")
        
        if self.model is None:
            print("⚠️ Model not trained. Using heuristic fallback.")
            return self._heuristic_fallback(ward_features_df)

        # Expected features for model
        feature_cols = ['rainfall_anomaly', 'pop_density_norm', 'dist_to_phc_km', 'facility_density_2km', 'cases_lag_1w']
        X = ward_features_df[feature_cols]
        
        # Predict Probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        # Generate SHAP explanations
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X)
        
        results = []
        for i, (idx, row) in enumerate(ward_features_df.iterrows()):
            score = probs[i]
            level = self._get_risk_level(score)
            
            # Extract top contributing features from SHAP
            row_shap = shap_values[i]
            top_feature_idx = np.argmax(np.abs(row_shap))
            top_feature = feature_cols[top_feature_idx]
            impact = "positive" if row_shap[top_feature_idx] > 0 else "negative"
            
            insight = f"Primary Driver: {top_feature.replace('_', ' ').title()} ({impact} impact)"
            
            results.append({
                "ward_id": row['ward_id'],
                "risk_score": round(float(score), 3),
                "risk_level": level,
                "insight": insight,
                "timestamp": datetime.now().isoformat()
            })
            
        return pd.DataFrame(results)

    def train_pilot_model(self, data_df):
        """
        Trains an XGBoost classifier with binary targets (High Case Risk).
        """
        print("🏋️ Training interpretable ward-risk model (Chennai Segment)...")
        
        feature_cols = ['rainfall_anomaly', 'pop_density_norm', 'dist_to_phc_km', 'facility_density_2km', 'cases_lag_1w']
        
        # Simulated target for training demo if not present
        if 'target' not in data_df.columns:
            # Simple rule-based target for pilot demonstration
            data_df['target'] = (data_df['cases_lag_1w'] > 5).astype(int)
            
        X = data_df[feature_cols]
        y = data_df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        clf.fit(X_train, y_train)
        
        self.model = clf
        joblib.dump(self.model, self.model_path)
        print(f"✅ Model saved to {self.model_path}")

    def _get_risk_level(self, score):
        if score > 0.8: return "CRITICAL"
        if score > 0.6: return "HIGH"
        if score > 0.4: return "MODERATE"
        return "LOW"

    def _heuristic_fallback(self, df):
        # Fallback scoring if model is absent
        results = []
        for _, row in df.iterrows():
            score = (row['rainfall_anomaly'] * 0.3 + 
                    (row.get('cases_lag_1w', 0) / 50) * 0.4 + 
                    (1 / (row.get('dist_to_phc_km', 1) + 1)) * 0.3)
            score = min(max(score, 0), 1)
            level = self._get_risk_level(score)
            results.append({
                "ward_id": row['ward_id'],
                "risk_score": round(score, 3),
                "risk_level": level,
                "insight": "Heuristic Calculation (Model Pending)",
                "timestamp": datetime.now().isoformat()
            })
        return pd.DataFrame(results)
