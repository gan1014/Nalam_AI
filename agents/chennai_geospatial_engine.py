
import pandas as pd
import json
import geopandas as gpd
from shapely.ops import nearest_points

class ChennaiGeospatialEngine:
    """
    MODULE A — GEO-SPATIAL ENGINE
    MODULE B — HEALTH ACCESS ANALYSIS
    """
    
    def __init__(self, wards_path):
        try:
            try:
                self.wards_gdf = gpd.read_file(wards_path)
            except Exception:
                # JSON Fallback for broken fiona/gdal installs
                with open(wards_path, "r") as f:
                    data = json.load(f)
                self.wards_gdf = gpd.GeoDataFrame.from_features(data["features"])
                self.wards_gdf.set_crs("EPSG:4326", inplace=True)
            
            # Standardize Identifiers (Step 1)
            # Use 'Ward_No' if present, and ensure it's integer
            if 'Ward_No' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['Ward_No'].astype(int)
            elif 'WARD_NO' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['WARD_NO'].astype(int)
            else:
                self.wards_gdf['ward_id'] = self.wards_gdf.index + 1
            
            # Print columns for debugging
            print(f"✅ Initialized Wards with columns: {list(self.wards_gdf.columns)}")
            
            # Ensure projected CRS for distance calculations (UTM 44N for Chennai)
            self.wards_proj = self.wards_gdf.to_crs(epsg=32644)
            print(f"✅ Projected Wards columns: {list(self.wards_proj.columns)}")
        except Exception as e:
            print(f"❌ Error loading Chennai Wards: {e}")
            self.wards_gdf = None

    def calculate_health_access(self, health_facilities_df):
        """
        Compute: 
        1. Distance to nearest facility (PHC/Hospital) in KM.
        2. Facility density (count within 2km radius).
        """
        if self.wards_gdf is None or health_facilities_df.empty:
            return pd.DataFrame()
            
        # Convert facilities to GDF (Step 2)
        facilities_gdf = gpd.GeoDataFrame(
            health_facilities_df, 
            geometry=gpd.points_from_xy(health_facilities_df.longitude, health_facilities_df.latitude),
            crs="EPSG:4326"
        ).to_crs(epsg=32644)
        
        # Remove ward_id from facilities if present to avoid collision in sjoin
        if 'ward_id' in facilities_gdf.columns:
            facilities_gdf = facilities_gdf.drop(columns=['ward_id'])
        
        # Step 3.1: Nearest Distance (Centroid-based)
        wards_temp = self.wards_proj.copy()
        # Explicitly ensure geometry is set
        if wards_temp.geometry.name != 'geometry':
            wards_temp = wards_temp.set_geometry('geometry')
            
        wards_temp['centroid_geom'] = wards_temp.geometry.centroid
        
        # Use centroid for distance to represent "Average Access"
        centroids_gdf = wards_temp.set_geometry('centroid_geom')
        # Rename to 'geometry' for standard spatial operations if needed, 
        # but sjoin_nearest should work with set_geometry.
        # Let's be explicit and use a gdf with only 'geometry'
        centroids_gdf = centroids_gdf[['ward_id', 'centroid_geom']].rename(columns={'centroid_geom': 'geometry'})
        centroids_gdf = centroids_gdf.set_geometry('geometry')
        
        # sjoin_nearest requires both to be GeoDataFrames
        # This will provide 'dist_m' column
        nearest = gpd.sjoin_nearest(centroids_gdf, facilities_gdf, distance_col="dist_m", how="left")
        
        # Group by ward_id and take min distance (in case of multiple matches)
        access_results = nearest.groupby('ward_id')['dist_m'].min().reset_index()
        access_results['dist_to_phc_km'] = access_results['dist_m'] / 1000.0
        
        # Step 3.2: Facility Density (2km radius)
        # Count facilities within 2000m of centroid
        buffer_2km = centroids_gdf.copy()
        buffer_2km['geometry'] = buffer_2km.geometry.buffer(2000)
        
        # Ensure facilities_gdf has geometry active
        if facilities_gdf.geometry.name != 'geometry':
            facilities_gdf = facilities_gdf.set_geometry('geometry')

        density_join = gpd.sjoin(facilities_gdf, buffer_2km[['ward_id', 'geometry']], how="inner", predicate="within")
        
        if not density_join.empty:
            density_counts = density_join.groupby('ward_id').size().reset_index(name='facility_density_2km')
            access_results = access_results.merge(density_counts, on='ward_id', how='left').fillna(0)
        else:
            access_results['facility_density_2km'] = 0
            
        print(f"📏 Computed health access for {len(access_results)} wards.")
        return access_results[['ward_id', 'dist_to_phc_km', 'facility_density_2km']]

    def get_ward_map_data(self):
        return self.wards_gdf
