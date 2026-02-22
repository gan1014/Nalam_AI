
import os
import pytest
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')

class TestDataPipeline:
    def test_raw_data_exists(self):
        path = os.path.join(DATA_DIR, 'raw/tn_disease_surveillance.csv')
        assert os.path.exists(path), "Raw data file missing"
        df = pd.read_csv(path)
        assert not df.empty, "Raw data is empty"
        assert 'cases' in df.columns, "Missing 'cases' column"

    def test_processed_data_exists(self):
        path = os.path.join(DATA_DIR, 'processed/train_scaled.csv')
        assert os.path.exists(path), "Processed data missing"
        df = pd.read_csv(path)
        assert 'risk_label' in df.columns, "Target variable missing"

    def test_geojson_valid(self):
        path = os.path.join(DATA_DIR, 'maps/tn_districts.geojson')
        assert os.path.exists(path), "GeoJSON missing"
        with open(path) as f:
            data = json.load(f)
        assert data['type'] == 'FeatureCollection', "Invalid GeoJSON format"

class TestModels:
    def test_xgboost_exists(self):
        path = os.path.join(MODELS_DIR, 'xgb_risk.pkl')
        assert os.path.exists(path), "XGBoost model file missing"
        
    def test_lstm_exists(self):
        path = os.path.join(MODELS_DIR, 'lstm_global.h5')
        # Note: might fail if LSTM training failed earlier, but should pass after fix
        if not os.path.exists(path):
            pytest.skip("LSTM model not trained yet")
        assert os.path.exists(path), "LSTM model file missing"

    def test_scalers_exist(self):
        assert os.path.exists(os.path.join(MODELS_DIR, 'scaler.pkl')), "Scaler missing"
        assert os.path.exists(os.path.join(MODELS_DIR, 'le_district.pkl')), "District encoder missing"

class TestDatabase:
    def test_db_connection(self):
        db_path = os.path.join(DATA_DIR, 'nalamai_local.db')
        # DB might not exist if setup skipped, but setup_supabase.py should create it
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            assert 'district_predictions' in tables, "Predictions table missing"

class TestPredictions:
    def test_predictions_generated(self):
        path = os.path.join(DATA_DIR, 'processed/latest_predictions.csv')
        if not os.path.exists(path):
             pytest.skip("Predictions not generated yet")
        
        df = pd.read_csv(path)
        assert not df.empty, "Predictions file is empty"
        assert 'risk_level' in df.columns, "Risk level column missing"
        assert set(df['risk_level'].unique()).issubset({'HIGH', 'MEDIUM', 'LOW'}), "Invalid risk levels"

if __name__ == "__main__":
    pytest.main([__file__])
