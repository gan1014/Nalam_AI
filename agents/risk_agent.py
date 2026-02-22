
import numpy as np

class RiskIntelligenceAgent:
    """
    AGENT 5 — Risk Intelligence Agent
    Converts raw prediction scores into action-oriented Risk Categories.
    """
    
    @staticmethod
    def categorize_risk(score):
        """
        Maps 0-1 score to categories: LOW | WATCH | HIGH | CRITICAL
        """
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.60:
            return "HIGH"
        elif score >= 0.35:
            return "WATCH"
        else:
            return "LOW"
            
    def process_scores(self, df):
        """
        Expects df with 'risk_score' (0-1)
        """
        df['risk_level'] = df['risk_score'].apply(self.categorize_risk)
        
        # Detect trend acceleration (mock logic for now)
        # In real case, we'd compare with previous week
        df['trend_status'] = "STABLE"
        
        return df
