
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_PATH = "nalamai/data/nalamai_local.db"

def init_supabase():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        supabase.table("district_predictions").select("*").limit(1).execute()
        print("✅ Supabase connection successful.")
        return supabase
    except Exception as e:
        print(f"⚠️ Supabase connection failed: {e}")
        return None

def init_sqlite():
    print(f"🔄 Initializing SQLite at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS district_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TIMESTAMP,
            district TEXT,
            disease TEXT,
            risk_score REAL,
            risk_level TEXT,
            predicted_cases INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            district TEXT,
            disease TEXT,
            risk_level TEXT,
            status TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            keyword TEXT,
            count INTEGER,
            sentiment REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hospital_beds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            district TEXT,
            total_beds INTEGER,
            available_beds INTEGER,
            last_updated TIMESTAMP
        )
    """)
    
    # Pre-populate hospital beds
    cursor.execute("SELECT COUNT(*) FROM hospital_beds")
    if cursor.fetchone()[0] == 0:
        data = [
            ('Chennai', 5000, 1200), ('Coimbatore', 3000, 800), ('Madurai', 2500, 600),
            ('Salem', 2000, 450), ('Tiruchirappalli', 2200, 500), ('Tirunelveli', 1800, 400),
            ('Erode', 1500, 300), ('Vellore', 1600, 350), ('Thoothukudi', 1400, 250),
            ('Thanjavur', 1300, 200)
        ]
        cursor.executemany("INSERT INTO hospital_beds (district, total_beds, available_beds, last_updated) VALUES (?, ?, ?, CURRENT_TIMESTAMP)", data)
        conn.commit()
        print("✅ Hospital beds table populated.")
        
    conn.close()
    print("✅ SQLite setup complete.")

def main():
    if SUPABASE_URL and SUPABASE_KEY:
        client = init_supabase()
        if not client:
            init_sqlite()
    else:
        print("ℹ️ Supabase credentials not found in .env. Using SQLite.")
        init_sqlite()

if __name__ == "__main__":
    main()
