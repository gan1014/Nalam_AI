
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "nalamai/data/nalamai_local.db"

def get_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        return None

def insert_prediction(date, district, disease, risk_score, risk_level, predicted_cases):
    conn = get_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO district_predictions (date, district, disease, risk_score, risk_level, predicted_cases)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date, district, disease, risk_score, risk_level, predicted_cases))
        conn.commit()
    except Exception as e:
        print(f"❌ Insert Prediction Error: {e}")
    finally:
        conn.close()

def get_latest_predictions():
    conn = get_connection()
    if not conn: return pd.DataFrame()
    try:
        query = """
            SELECT date, district, disease, risk_score, risk_level, predicted_cases
            FROM district_predictions
            WHERE date = (SELECT MAX(date) FROM district_predictions)
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"❌ Get Predictions Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def log_alert(district, disease, risk_level, status, triggered_by="System", recommendations=""):
    conn = get_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts_log (timestamp, district, disease, risk_level, status, triggered_by, recommendations)
            VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
        """, (district, disease, risk_level, status, triggered_by, recommendations))
        conn.commit()
    except Exception as e:
        print(f"❌ Log Alert Error: {e}")
    finally:
        conn.close()

def get_hospital_data():
    conn = get_connection()
    if not conn: return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM hospital_beds", conn)
        return df
    except Exception as e:
        print(f"❌ Get Hospital Data Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def insert_social_signal(keyword, count, sentiment):
    conn = get_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO social_signals (timestamp, keyword, count, sentiment)
            VALUES (CURRENT_TIMESTAMP, ?, ?, ?)
        """, (keyword, count, sentiment))
        conn.commit()
    except Exception as e:
        print(f"❌ Insert Social Signal Error: {e}")
    finally:
        conn.close()

def log_audit(ward_id, action, user_role, details="", user_id=0, ip_address="127.0.0.1"):
    """Legacy audit function - redirects to log_event for reverse compatibility"""
    log_event("GUEST", user_role, "N/A", action, "Ward", str(ward_id), details, ip_address)

def init_audit_db():
    conn = get_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS government_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                role TEXT,
                session_id TEXT,
                action TEXT,
                target_type TEXT,
                target_id TEXT,
                details TEXT,
                timestamp DATETIME,
                ip_address TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                embedding_blob BLOB,
                created_at DATETIMEDEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp DATETIME,
                status TEXT,
                ip_address TEXT,
                device_type TEXT,
                similarity_score REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"❌ Init Audit DB Error: {e}")
    finally:
        conn.close()

def init_face_db():
    # This is now integrated into init_audit_db for convenience, 
    # but we keep the name for consistency with plan if needed.
    init_audit_db()

def save_face_embedding(user_id, embedding_blob):
    conn = get_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_face_embeddings (user_id, embedding_blob)
            VALUES (?, ?)
        """, (user_id, embedding_blob))
        conn.commit()
    except Exception as e:
        print(f"❌ Save Face Embedding Error: {e}")
    finally:
        conn.close()

def get_face_embeddings(user_id):
    conn = get_connection()
    if not conn: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding_blob FROM user_face_embeddings WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"❌ Get Face Embeddings Error: {e}")
        return []
    finally:
        conn.close()

def log_face_verification(user_id, status, ip_address="127.0.0.1", device_type="Unknown", similarity_score=0.0):
    conn = get_connection()
    if not conn: return
    try:
        from datetime import datetime, timedelta, timezone
        ist = timezone(timedelta(hours=5, minutes=30))
        ts_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO face_verification_logs (user_id, timestamp, status, ip_address, device_type, similarity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, ts_ist, status, ip_address, device_type, similarity_score))
        conn.commit()
        
        # Also log to main government audit log for unified visibility
        username = "Unknown"
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if user: username = user[0]
        
        log_event(username, "USER", "N/A", f"FACE_VERIFY_{status}", "Biometrics", str(user_id), f"Score: {similarity_score}", ip_address)
    except Exception as e:
        print(f"❌ Log Face Verification Error: {e}")
    finally:
        conn.close()

def log_event(username, role, session_id, action, target_type, target_id, details="", ip_address="127.0.0.1"):
    """
    Government-Grade Audit Logger
    Captures WHO, WHAT, WHERE, WHEN for accountability.
    """
    conn = get_connection()
    if not conn: return
    try:
        # Get server-side time in IST
        from datetime import datetime, timedelta, timezone
        ist = timezone(timedelta(hours=5, minutes=30))
        ts_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO government_audit_logs (
                username, role, session_id, action, target_type, target_id, details, timestamp, ip_address
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, role, session_id, action, target_type, target_id, details, ts_ist, ip_address))
        conn.commit()
    except Exception as e:
        print(f"❌ Log Event Error: {e}")
    finally:
        conn.close()

def get_audit_logs(limit=500, user_filter=None, action_filter=None):
    conn = get_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT * FROM government_audit_logs"
        params = []
        conditions = []
        if user_filter:
            conditions.append("username = ?")
            params.append(user_filter)
        if action_filter:
            conditions.append("action = ?")
            params.append(action_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        print(f"❌ Get Audit Logs Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Initialize on import
init_audit_db()
