
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
RAW_DATA_PATH = 'nalamai/data/raw/tn_disease_surveillance.csv'
PROCESSED_DATA_DIR = 'nalamai/data/processed'
MODELS_DIR = 'nalamai/models'

def create_lag_features(df, lags=[1, 2, 4, 8]):
    for lag in lags:
        df[f'cases_lag{lag}'] = df.groupby(['district', 'disease'])['cases'].shift(lag)
    return df

def create_rolling_features(df, window=4):
    grouped = df.groupby(['district', 'disease'])['cases']
    df[f'cases_roll{window}_mean'] = grouped.transform(lambda x: x.rolling(window).mean())
    df[f'cases_roll{window}_max'] = grouped.transform(lambda x: x.rolling(window).max())
    df[f'cases_roll{window}_std'] = grouped.transform(lambda x: x.rolling(window).std())
    return df

def create_cyclic_features(df):
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['date'].dt.isocalendar().week / 53)
    df['week_cos'] = np.cos(2 * np.pi * df['date'].dt.isocalendar().week / 53)
    return df

def create_interaction_features(df):
    df['rain_x_humidity'] = df['rainfall_mm'] * df['humidity']
    df['temp_x_humidity'] = df['temp_max'] * df['humidity']
    return df

def create_binary_flags(df):
    df['high_rain_flag'] = (df['rainfall_mm'] > 100).astype(int)
    df['extreme_rain_flag'] = (df['rainfall_mm'] > 200).astype(int)
    df['hot_weather_flag'] = (df['temp_max'] > 35).astype(int)
    return df

def create_risk_label(df):
    # Calculate 75th percentile threshold per disease
    thresholds = df.groupby('disease')['cases'].quantile(0.75).to_dict()
    df['risk_label'] = df.apply(lambda row: 1 if row['cases'] > thresholds[row['disease']] else 0, axis=1)
    return df, thresholds

def main():
    print("Starting data preprocessing...")
    
    # Load data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"Error: {RAW_DATA_PATH} not found. Run generate_data.py first.")
        return

    # Sort
    df = df.sort_values(['district', 'disease', 'date']).reset_index(drop=True)
    
    # Feature Engineering
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_cyclic_features(df)
    df = create_interaction_features(df)
    df = create_binary_flags(df)
    
    # Create Target
    df, thresholds = create_risk_label(df)
    
    # Drop rows with NaN due to lags/rolling
    df = df.dropna().reset_index(drop=True)
    
    # Encoding
    le_district = LabelEncoder()
    df['district_encoded'] = le_district.fit_transform(df['district'])
    
    le_disease = LabelEncoder()
    df['disease_encoded'] = le_disease.fit_transform(df['disease'])
    
    # Scaling
    feature_cols = [col for col in df.columns if col not in ['date', 'district', 'disease', 'cases', 'risk_label']]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save Artifacts
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df.to_csv(f'{PROCESSED_DATA_DIR}/train_scaled.csv', index=False)
    
    # Save encoders and scaler
    joblib.dump(le_district, f'{MODELS_DIR}/le_district.pkl')
    joblib.dump(le_disease, f'{MODELS_DIR}/le_disease.pkl')
    joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
    joblib.dump(thresholds, f'{MODELS_DIR}/thresholds.pkl')
    
    print(f"✅ Data Preprocessing Complete.")
    print(f"Dataset Shape: {df.shape}")
    print(f"Risk Distribution:\n{df['risk_label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
