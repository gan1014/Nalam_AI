
import pandas as pd

class RecommendationEngine:
    """
    AGENT 6 — Recommendation Engine
    Translates predictions into public-health actions.
    """
    
    @staticmethod
    def get_recommendations(risk_level, disease):
        recommendations = {
            "CRITICAL": [
                "🚨 IMMEDIATE ACTION: Dispatch vector control teams to hotspots within 24 hours.",
                "🏥 Capacity Surge: Prepare District General Hospital and PHCs for immediate case influx.",
                "📢 Emergency Advisory: Launch intensive community awareness campaign in affected blocks.",
                "🔍 Active Surveillance: Initiate door-to-door fever surveillance in priority wards."
            ],
            "HIGH": [
                "⚠️ Priority Action: Begin targeted fogging and source reduction in high-risk zones.",
                "🏥 Preparedness: Ensure adequate stock of essential medicines and diagnostic kits.",
                "📈 Enhanced Monitoring: Daily reporting of new cases from local clinics required.",
                "📢 Local Advisory: Issue health advisories through local media and community leaders."
            ],
            "WATCH": [
                "👀 Surveillance: Monitor case trends closely for next 7 days.",
                "🧹 Prevention: Routine sanitation and vector control measures in identified areas.",
                "📋 Verification: Cross-verify reported cases with clinical samples."
            ],
            "LOW": [
                "✅ Normal Operations: Continue routine surveillance.",
                "📊 Data Logging: Maintain weekly reporting cycle."
            ]
        }
        
        # Specific recommendations for Tamil Nadu monsoon focus
        tn_monsoon_extras = {
            "Dengue": "Ensure drain clearance and stagnant water removal (NE Monsoon prep).",
            "Cholera": "Audit chlorination levels in community water tanks.",
            "Leptospirosis": "Distribute prophylaxis to workers in flooded agriculture zones."
        }
        
        recs = recommendations.get(risk_level, recommendations["LOW"])
        if disease in tn_monsoon_extras and risk_level in ["CRITICAL", "HIGH"]:
            recs.append(f"🌧️ Monsoon Specific: {tn_monsoon_extras[disease]}")
            
        return recs

    def process_predictions(self, predictions_df):
        """
        Takes predictions and adds recommendation strings.
        """
        def format_recs(row):
            recs = self.get_recommendations(row['risk_level'], row['disease'])
            return " | ".join(recs)
            
        predictions_df['recommendations'] = predictions_df.apply(format_recs, axis=1)
        return predictions_df
