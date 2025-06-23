import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'lib'))

import pandas as pd 
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

#Local import
from utils.date import extract_date_components

def merge_by_month(source_df,inc_df,radius):
    # Create list of months (1-12)
    months = list(range(1, 13))
    
    source_df['HAS_FIRE_IN_MONTH']=False #sets all values to False by default
    print(f"Starting incident matching")
    print(f"['HAS_FIRE_IN_MONTH'] contains {source_df['HAS_FIRE_IN_MONTH'].value_counts()}")
    source_gdf = gpd.GeoDataFrame(
                    source_df,
                    geometry=gpd.points_from_xy(source_df["LONGITUDE"].astype(float),
                                                source_df["LATITUDE"].astype(float)),
                    crs="EPSG:4326"
                )
    
    incident_gdf = gpd.GeoDataFrame(
        inc_df,
        geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
        crs="EPSG:4326"
    )

    # --- Project both to meters for spatial operations ---

    incident_gdf = incident_gdf.to_crs(epsg=32188)
    source_gdf = source_gdf.to_crs(epsg=32188)


    for m in months:
        
        monthly_incidents_gdf=incident_gdf[incident_gdf['CREATION_DATE_TIME'].dt.month==m].copy()
        print(monthly_incidents_gdf.head(5))
        print(f"month {m} : {monthly_incidents_gdf.shape}")
        
        
        # --- Buffer fire incidents by radius meters ---
        monthly_incidents_gdf["buffer"] = monthly_incidents_gdf.geometry.buffer(radius)
        incident_buffer_gdf = monthly_incidents_gdf.set_geometry("buffer")
        
        # --- Spatial join: find properties within 100m of a fire incident ---
        joined = gpd.sjoin(source_gdf, incident_buffer_gdf, predicate='within', how='inner')
        # print(f"joined length : {joined.shape}")
        # --- Use unique matched ID_UEV set to mark fires ---
        matched_ids = set(joined["ID_UEV"])
        # assign HAS_FIRE_IN_MONTH VALUE if there has been a fire in this month
        source_df["HAS_FIRE_IN_MONTH"] = (source_df["ID_UEV"].isin(matched_ids) & (source_df['month']==m)) | (source_df['HAS_FIRE_IN_MONTH']==True)
        print(f"Finished processing month {m}")
        print(f"['HAS_FIRE_IN_MONTH'] contains {source_df['HAS_FIRE_IN_MONTH'].value_counts()}")