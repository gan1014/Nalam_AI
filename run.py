
import os
import sys
import time
import subprocess
import threading

def run_command(command, description):
    print(f"\n🚀 {description}...")
    try:
        # Run command and wait for it to complete
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"✅ {description} Complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} Failed: {e}")
        return False

def main():
    print("==================================================")
    print("   Nalam AI · நலம் AI — Automated Deployment")
    print("   Tamil Nadu Clinical Surveillance System")
    print("==================================================")
    
    # 1. Data Generation
    if not run_command("python ml/generate_data.py", "Data Generation"): return

    # 2. Preprocessing
    if not run_command("python ml/preprocess.py", "Data Preprocessing"): return

    # 3. Model Training
    if not run_command("python ml/train_xgb.py", "Training XGBoost Model"): return
    if not run_command("python ml/train_lstm.py", "Training LSTM Model"): return

    # 4. SHAP Explanation
    if not run_command("python ml/shap_explain.py", "Generating Explainability Plots"): return

    # 5. Database Setup
    if not run_command("python backend/setup_supabase.py", "Database Initialization"): return

    if not run_command("python -m ml.chennai_pilot_build", "Building Chennai Ward-Level Pilot"): return
    
    if not run_command("python -m agents.intelligence_engine", "Running NalamAI Intelligence (Agents 5 & 6)"): return

    # 7. Launch Dashboard & API
    print("\n✅ All Pipeline Steps Complete!")
    print("\n==================================================")
    print("   LAUNCHING SERVICES")
    print("==================================================")
    print("1. Streamlit Dashboard: http://localhost:8501")
    print("2. FastAPI Backend: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop servers.")
    
    try:
        # Run Streamlit in background
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py"], shell=True)
        
        # Run FastAPI in foreground (or vice versa)
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"], shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")

if __name__ == "__main__":
    main()
