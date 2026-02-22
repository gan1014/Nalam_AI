
import os
import yaml
import json
import datetime
import pandas as pd

class DataTrustAgent:
    """
    Combines AGENT 1 (Acquisition) & AGENT 2 (Validation)
    Enforces Data Trust Policy via Whitelist.
    """
    
    def __init__(self, whitelist_path="config/data_whitelist.yaml"):
        self.whitelist_path = whitelist_path
        with open(whitelist_path, 'r') as f:
            self.whitelist = yaml.safe_load(f)
            
    def verify_source(self, source_name):
        approved = self.whitelist.get('approved_sources', [])
        source = next((s for s in approved if s['source_name'] == source_name), None)
        
        if not source:
            raise ValueError(f"❌ DATA TRUST VIOLATION: Source '{source_name}' is NOT in the whitelist.")
        
        if source['approval_status'] != 'allowed':
            raise ValueError(f"❌ DATA TRUST VIOLATION: Source '{source_name}' is BLOCKED.")
            
        print(f"✅ Source verified: {source_name}")
        return source

    def log_lineage(self, step, metadata):
        lineage_path = "logs/data_lineage.json"
        os.makedirs(os.path.dirname(lineage_path), exist_ok=True)
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "metadata": metadata
        }
        
        if os.path.exists(lineage_path):
            with open(lineage_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(entry)
        with open(lineage_path, 'w') as f:
            json.dump(logs, f, indent=2)

    def validate_dataset(self, df, source_name):
        """
        Check schema and flag anomalies.
        """
        report = {
            "source": source_name,
            "rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "anomalies": []
        }
        
        # Simple anomaly detection: Negative counts
        if 'cases' in df.columns:
            negatives = df[df['cases'] < 0]
            if not negatives.empty:
                report['anomalies'].append(f"Found {len(negatives)} negative case counts.")
                
        self.log_lineage("VALIDATION", report)
        return report
