import pandas as pd
import numpy as np
from datetime import datetime

class ChennaiFeatureEngineer:
    def __init__(self, surveillance_path):
        self.surveillance_path = surveillance_path

    def prepare_chennai_features(self, target_disease='Dengue'):
        """
        Extracts Chennai district data, calculates lags and anomalies.
        """
        df = pd.read_csv(self.surveillance_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for Chennai and target disease
        chennai_df = df[(df['district'] == 'Chennai') & (df['disease'] == target_disease)].copy()
        chennai_df = chennai_df.sort_values('date')
        
        # 1. Historical Lags (Step 4.2)
        chennai_df['cases_lag_1w'] = chennai_df['cases'].shift(1)
        chennai_df['cases_lag_4w'] = chennai_df['cases'].shift(4)
        
        # 2. Environmental Signals (Step 4.1)
        # Rainfall Anomaly (Current - 4 week rolling mean)
        chennai_df['rain_roll_4w'] = chennai_df['rainfall_mm'].rolling(window=4).mean()
        chennai_df['rainfall_anomaly'] = chennai_df['rainfall_mm'] - chennai_df['rain_roll_4w']
        
        # Temperature signals
        chennai_df['temp_max_roll_4w'] = chennai_df['temp_max'].rolling(window=4).mean()
        
        # Drop initial NaNs from lagging
        chennai_df = chennai_df.dropna()
        
        return chennai_df

if __name__ == "__main__":
    fe = ChennaiFeatureEngineer("data/raw/tn_disease_surveillance.csv")
    features = fe.prepare_chennai_features()
    print(f"Engineered {len(features)} historical time-steps for Chennai.")
    print(features[['date', 'cases', 'cases_lag_1w', 'rainfall_anomaly']].tail())
