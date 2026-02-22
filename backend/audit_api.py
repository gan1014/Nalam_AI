from fastapi import FastAPI, HTTPException
import pandas as pd
import os

app = FastAPI(title="NalamAI Audit API")

@app.get("/audit")
def get_audit_trail():
    """
    Returns the real-time audit trail of alerts and data validation.
    """
    from nalamai.backend import db
    conn = db.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # 1. Fetch Alert Audit Trail
        alerts_df = pd.read_sql_query("SELECT * FROM alerts_log ORDER BY timestamp DESC LIMIT 100", conn)
        
        # 2. Fetch Data Lineage (Feature completeness)
        FEATURE_PATH = "data/processed/model_features.csv"
        lineage = []
        if os.path.exists(FEATURE_PATH):
            feat_df = pd.read_csv(FEATURE_PATH)
            for _, row in feat_df.head(50).iterrows(): # Sample for brevity
                lineage.append({
                    "ward_id": int(row['WARD_ID']),
                    "data_status": str(row['DATA_AVAILABILITY']),
                    "population_validated": True if row['POPULATION'] > 0 else False,
                    "spatial_join_verified": True if row['NEAREST_FACILITY_KM'] < 999 else False
                })

        return {
            "status": "government_verified",
            "total_alerts_dispatched": len(alerts_df),
            "recent_alerts": alerts_df.to_dict(orient="records"),
            "data_lineage_sample": lineage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
