# Ugly script to run all dataprep and data merge scripts
import os

steps = [
    os.path.join('dataprep','evaluation_fonciere.py') #output eval_cleaned_feat_eng.csv
    ,os.path.join('dataprep','interventions_HAS_FIRE_binary_analysis.py') #output interventions_cleaned_with_has_fire.csv
    ,os.path.join('dataprep','main_evaluation_feat_eng.py') #merge addresses +fire +date into ./datasets/cleaned/evaluation_fire_coordinates_date_feat_eng_2.csv
    ,os.path.join('dataprep','dense_panel_building_month.py') #build panel and save to cleaned/building_month_fire_panel_feat_eng.csv
]
for step in steps:
    print(f"Executing {step}")
    os.system(f"python {step}")  
    print("\n\n")
    