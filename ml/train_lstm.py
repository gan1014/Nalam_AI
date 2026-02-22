
import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available (likely due to Windows long path restrictions). Mocking LSTM global model.")
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Paths
PROCESSED_DATA_PATH = 'nalamai/data/processed/train_scaled.csv'
MODEL_PATH = 'nalamai/models/lstm_global.h5'

SEQUENCE_LENGTH = 8
BATCH_SIZE = 64
EPOCHS = 5

def create_sequences(df, sequence_length):
    X, y = [], []
    # Group by district and disease to ensure temporal consistency
    for _, group in df.groupby(['district_encoded', 'disease_encoded']):
        data = group.drop(['date', 'district', 'disease', 'cases', 'risk_label'], axis=1).values
        target = group['cases'].values
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(target[i+sequence_length])
            
    return np.array(X), np.array(y)

def build_model(input_shape):
    if not TF_AVAILABLE:
        return None
        
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("Training Global LSTM Forecasting Model...")
    
    # Load data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {PROCESSED_DATA_PATH} not found.")
        return

    # Create sequences
    X, y = create_sequences(df, SEQUENCE_LENGTH)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build Model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    if not TF_AVAILABLE:
        # Save mock file
        with open(MODEL_PATH, 'w') as f:
            f.write("mock_lstm_h5")
        print(f"✅ Mock LSTM Model saved to {MODEL_PATH}")
        return

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save
    model.save(MODEL_PATH)
    print(f"✅ LSTM Model saved to {MODEL_PATH}")
    print(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
