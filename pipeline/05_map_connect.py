import pandas as pd
import geopandas as gpd
import json
import os

def connect_map():
    print("🌍 Phase 8: Linking Model Results to Map...")
    
    WARDS_GEOJSON = "data/processed/wards.geojson"
    SCORES_PATH = "outputs/ward_risk_scores.csv"
    OUTPUT_PATH = "data/chennai/processed/chennai_pilot_intelligence.geojson"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Load Data
    with open(WARDS_GEOJSON, 'r') as f:
        data = json.load(f)
    wards = gpd.GeoDataFrame.from_features(data["features"])
    wards.set_crs("EPSG:4326", inplace=True)
    wards.columns = [c.upper() for c in wards.columns]
    if 'geometry' in wards.columns: wards.set_geometry('geometry', inplace=True)

    scores = pd.read_csv(SCORES_PATH)

    # 2. Merge (Task 1: Use LEFT JOIN)
    merged = wards.merge(scores, on='WARD_ID', how='left')
    merged = gpd.GeoDataFrame(merged, geometry='GEOMETRY', crs="EPSG:4326")

    # 3. Standardize Columns for app.py Compatibility
    merged['ward_id'] = merged['WARD_ID']
    merged['risk_score'] = merged['RISK_SCORE']
    merged['dist_to_phc_km'] = merged['NEAREST_FACILITY_KM']
    merged['status'] = merged['DATA_AVAILABILITY'].fillna("DATA_NOT_AVAILABLE")
    
    # Categorize Risk Level
    def categorize(row):
        if row['status'] == "DATA_NOT_AVAILABLE": return "DATA_PENDING"
        # VALIDATED and VALIDATED_ESTIMATE both get colored scores
        score = row['risk_score']
        if score > 0.8: return "CRITICAL"
        if score > 0.6: return "HIGH"
        if score > 0.4: return "MODERATE"
        return "LOW"
    
    merged['risk_level'] = merged.apply(categorize, axis=1)
    
    # Create human-readable insights (Task 6)
    def make_insight(row):
        if row['status'] == "DATA_NOT_AVAILABLE":
            return f"Ward {int(row['WARD_ID'])}: Case Data Pending Validation"
        
        driver = str(row['TOP_RISK_DRIVER']).replace('_', ' ').title()
        prefix = ""
        if row['status'] == "VALIDATED_ESTIMATE":
            prefix = "[Estimate] "
        
        return f"{prefix}Ward {int(row['WARD_ID'])} Primary Risk Driver: {driver}. Score: {row['risk_score']:.2f}"

    merged['insight'] = merged.apply(make_insight, axis=1)

    # Task 8: Generate Coverage Report
    total_wards = len(merged)
    # Task 11: Analysed wards includes both direct and estimated validation
    matched_wards = len(merged[merged['status'].isin(["VALIDATED", "VALIDATED_ESTIMATE"])])
    missing_wards = total_wards - matched_wards
    completeness = (matched_wards / total_wards) * 100 if total_wards > 0 else 0
    
    REPORT_PATH = "ward_coverage_report.txt"
    with open(REPORT_PATH, 'w') as f:
        f.write("--- CHENNAI WARD COVERAGE REPORT ---\n")
        f.write(f"Total wards in boundary: {total_wards}\n")
        f.write(f"Analysed (Validated) wards: {matched_wards}\n")
        f.write(f"Missing data wards: {missing_wards}\n")
        f.write(f"Data completeness %: {completeness:.2f}%\n")
    print(f"📄 Coverage report generated: {REPORT_PATH}")

    # Task 2: Mismatch report
    unmatched = merged[merged['status'] == "DATA_NOT_AVAILABLE"]['WARD_ID']
    unmatched.to_csv("unmatched_wards.csv", index=False)

    # 4. Save to final GeoJSON (Task 1, 6)
    # Preservation of all polygons is guaranteed by LEFT JOIN
    merged.to_file(OUTPUT_PATH, driver='GeoJSON')
    print(f"✅ Phase 8 Complete. Enriched map saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    connect_map()
