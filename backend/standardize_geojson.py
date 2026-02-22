import json
import os
import pandas as pd

def standardize_properties(feature, source_type, risk_data_map=None):
    props = feature.get("properties", {})
    # Start with a copy to preserve existing fields
    std_props = props.copy()
    
    # 1. Map core GIS keys
    if source_type in ["wards", "chennai_wards_standardized"]:
        # Try different naming variations for core IDs
        std_props["ward_id"] = int(props.get("ward_id", props.get("WARD_ID", props.get("Ward_No", props.get("WARD_NO", 0)))))
        std_props["ward_no"] = int(props.get("ward_no", props.get("Ward_No", props.get("WARD_NO", props.get("WARD_ID", 0)))))
        std_props["ward_name"] = str(props.get("ward_name", props.get("WARD_NAME", props.get("WARD_NAME_CLEAN", "")))).strip().lower()
        std_props["zone_id"] = str(props.get("zone_id", props.get("Zone_No", props.get("ZONE_NO", "")))).strip()
        std_props["zone_name"] = str(props.get("zone_name", props.get("Zone_Name", props.get("ZONE_NAME", "")))).strip().lower()
        std_props["population"] = int(props.get("population", props.get("POPULATION", 0)))
        std_props["area_sqkm"] = float(props.get("area_sqkm", props.get("AREA", props.get("WARD_AREA_KM2", 0.0))))

    # 2. Add/Override with Analytical Data from CSV if available
    ward_id = std_props.get("ward_id")
    if risk_data_map and ward_id in risk_data_map:
        csv_data = risk_data_map[ward_id]
        for k, v in csv_data.items():
            # Standardize common keys to lowercase for app compatibility
            target_key = k.lower()
            # Handle special mappings if needed
            if target_key == "data_availability": target_key = "status"
            
            # Don't overwrite ward_id/no unless missing
            if target_key in ["ward_id", "ward_no"] and std_props.get(target_key):
                continue
            
            std_props[target_key] = v

    # 3. Final normalization and dynamic calculations
    for key in list(std_props.keys()):
        if key.lower() in ["risk_score", "risk_level", "insight", "status", "dist_to_phc_km"]:
            val = std_props.pop(key)
            std_props[key.lower()] = val

    # Dynamic Risk Level Calculation if score exists
    if "risk_score" in std_props:
        score = float(std_props["risk_score"])
        if score > 0.8: std_props["risk_level"] = "CRITICAL"
        elif score > 0.6: std_props["risk_level"] = "HIGH"
        elif score > 0.4: std_props["risk_level"] = "MODERATE"
        else: std_props["risk_level"] = "LOW"

    # Basic fallback/cleanup
    if not std_props.get("ward_name"):
        std_props["ward_name"] = f"ward_{std_props.get('ward_no', 'unknown')}"

    feature["properties"] = std_props
    return feature

def process_file(input_path, output_path, source_type, risk_data_map=None):
    print(f"Processing {input_path}...")
    try:
        if not os.path.exists(input_path):
            print(f"Skipping {input_path} (not found)")
            return

        with open(input_path, 'r') as f:
            data = json.load(f)
            
        data["features"] = [standardize_properties(f, source_type, risk_data_map) for f in data.get("features", [])]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    base_dir = "d:/NALAM_AI"
    
    # Load risk data map if it exists
    risk_csv = os.path.join(base_dir, "outputs/ward_risk_scores.csv")
    risk_data_map = {}
    if os.path.exists(risk_csv):
        print(f"Loading analytical data from {risk_csv}...")
        df_risk = pd.read_csv(risk_csv)
        # Handle different ID casings in CSV
        id_col = "WARD_ID" if "WARD_ID" in df_risk.columns else "ward_id"
        risk_data_map = df_risk.set_index(id_col).to_dict('index')
    
    files_to_process = [
        ("data/processed/wards.geojson", "data/processed/standardized/wards.geojson", "wards"),
        ("data/processed/chennai_wards_standardized.geojson", "data/processed/standardized/chennai_wards_standardized.geojson", "chennai_wards_standardized"),
        ("data/chennai/processed/chennai_pilot_intelligence.geojson", "data/processed/standardized/chennai_pilot_intelligence.geojson", "chennai_wards_standardized")
    ]
    
    for in_file, out_file, s_type in files_to_process:
        process_file(
            os.path.join(base_dir, in_file),
            os.path.join(base_dir, out_file),
            s_type,
            risk_data_map
        )
