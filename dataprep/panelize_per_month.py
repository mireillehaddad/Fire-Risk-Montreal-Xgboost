import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'lib'))

import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

#Local import
from utils.date import extract_date_components
from utils.merge_fire import merge_by_month




SOURCE_FILE = os.path.join('.','datasets','merged','eval_with_coord.csv')
DESTINATION_FILE = os.path.join('.','datasets','merged','base_panel_month.csv')
INTERVENTIONS_FILE_PATH = os.path.join('.','datasets','cleaned','interventions_cleaned.csv')

print("Loading incidents before 2024... ")
incidents=pd.read_csv(INTERVENTIONS_FILE_PATH,parse_dates=['CREATION_DATE_TIME'])
incidents = extract_date_components(incidents,'CREATION_DATE_TIME',date_format= '%Y-%m-%d %H:%M:%S')
print(incidents.columns)
incidents=incidents[incidents['DESCRIPTION_GROUPE']=='INCENDIE']
print(f"Incident categories kept: {incidents['DESCRIPTION_GROUPE'].unique()}")
# -- Project both to meters for spatial operations ---

def merge_incidents(incidents,output_file):
    print(f"Loading dataset from {SOURCE_FILE}...")
    df = pd.read_csv(SOURCE_FILE)
    print(f"dataset loaded - shape {df.shape}")


    # Create list of months (1-12)
    months = list(range(1, 13))

    # Duplicate each row 12 times and add month column
    df = df.loc[df.index.repeat(12)].assign(month=np.tile(months, len(df))).reset_index(drop=True)




    merge_by_month(df,incidents,100)



    print(f"Saving file to {output_file}")
    print(f"Resulting dataframe's shape: {df.shape}")
    df.to_csv(output_file,index=False)

merge_incidents(incidents[incidents['CREATION_DATE_TIME_year']<2024],DESTINATION_FILE.replace('.csv','_train.csv'))
print(f"Train File saved")
merge_incidents(incidents[incidents['CREATION_DATE_TIME_year']>=2024],DESTINATION_FILE.replace('.csv','_test.csv'))
print(f"Test File saved")

#to merge to a single file all incidents, use this instead:
# merge_incidents(incidents,DESTINATION_FILE)
#print(f"Merged File saved")
