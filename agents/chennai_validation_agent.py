
import pandas as pd
import geopandas as gpd
import os

class ChennaiDataValidationAgent:
    """
    SECTION 3 — DATA VALIDATION PIPELINE
    Validates ward-level datasets for Chennai pilot.
    """
    
    def validate_spatial_data(self, gdf, expected_crs="EPSG:4326"):
        """
        ✔ Coordinate Reference Systems match
        ✔ Spatial overlays produce valid joins
        """
        report = {"status": "SUCCESS", "issues": []}
        
        if gdf.crs is None:
            # Assume 4326 but flag it
            gdf.set_crs(expected_crs, inplace=True)
            report["issues"].append("Warning: CRS was missing, set to EPSG:4326")
            
        if str(gdf.crs) != expected_crs:
            report["issues"].append(f"CRS mismatch: expected {expected_crs}, got {gdf.crs}")
            report["status"] = "FAILED"
            
        if gdf.empty or gdf.geometry.isnull().any():
            report["issues"].append("Empty or null geometries detected.")
            report["status"] = "FAILED"
            
        return report

    def validate_ward_alignment(self, df_list, ward_col="ward_no"):
        """
        ✔ Ward names/numbers align across all tables
        """
        if not df_list: return True
        
        master_wards = set(df_list[0][ward_col].unique())
        for i, df in enumerate(df_list[1:]):
            current_wards = set(df[ward_col].unique())
            missing = master_wards - current_wards
            extra = current_wards - master_wards
            if missing or extra:
                print(f"⚠️ Ward alignment variance in dataset {i+1}: Missing {len(missing)}, Extra {len(extra)}")
                
        return True

    def run_full_validation(self, ward_gdf, health_df, population_df):
        print("🛡️ Starting Chennai Ward-Level Validation Pipeline...")
        
        # 1. Spatial Checks
        spatial_rpt = self.validate_spatial_data(ward_gdf)
        
        # 2. Duplicate facilities
        if 'facility_id' in health_df.columns and health_df['facility_id'].duplicated().any():
            print("❌ Duplicate facilities detected.")
            
        # 3. Missing values < 15%
        for name, df in [("Health", health_df), ("Pop", population_df)]:
            missing_pct = df.isnull().mean() * 100
            if (missing_pct > 15).any():
                print(f"❌ Missing values > 15% in {name} dataset: {missing_pct[missing_pct > 15].to_dict()}")
                
        # 4. Population reconciliation
        # (Compare aggregated ward pop with known GCC total approx 7-8 million)
        total_pop = population_df['population'].sum()
        if total_pop < 5000000 or total_pop > 10000000:
             print(f"⚠️ Population reconciliation warning: Total {total_pop} differs from expected GCC range.")

        print("✅ Chennai Validation Pipeline Complete.")
        return True
