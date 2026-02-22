
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json

# Paths
PROCESSED_DATA_PATH = 'nalamai/data/processed/train_scaled.csv'
MODEL_PATH = 'nalamai/models/xgb_risk.pkl'
METRICS_PATH = 'nalamai/models/xgb_metrics.json'

def main():
    print("Training XGBoost Risk Classifier...")
    
    # Load data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {PROCESSED_DATA_PATH} not found. Run preprocess.py first.")
        return

    # Split
    X = df.drop(['risk_label', 'cases', 'date', 'district', 'disease'], axis=1)
    y = df['risk_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Class Imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Model
    xgb = XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='auc', random_state=42, n_jobs=-1
    )
    
    # Train
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"✅ Training Complete. Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(xgb, MODEL_PATH)
    
    metrics = {'accuracy': acc, 'auc': auc}
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
        
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
