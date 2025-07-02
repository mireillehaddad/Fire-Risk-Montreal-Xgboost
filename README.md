

# all-capstone-project-summer-2025-team-6
## Table of Contents
- [Getting started](#getting-started)
- [Data Pipeline Diagram](#data-pipeline-diagram)
- [How to Run the Data Pipeline](#how-to-run-the-data-pipeline)

## Getting started
1. Run the data pipeline to load, clean, impute, and add engineered features, then build the merged building x fire incident monthly panel.
```commandline
python ./datapipeline_panel_add_features.py
```
2. Train models :
- XGBoost on panel data
```commandline
python ./datamodel/xgboost_panel_with_feat.py
```
- Random Forest on panel data
```commandline
python ./datamodel/random_forest_from_panel_with_feat.py
```

Each of these will train and save the model,label encoder and feature set used as Pickle files and output a CSV file containing the test set (2024 data) with added columns for predicted result and predicted probability of being in "True" class.

3. datamodel/model_analysis_template.ipynb is a Jupyter notebook that can be used to load the saved models and exported test set for analysis.

4. datamodel/model_k_precision_visualizations.ipynb is a Jupyter notebook that can be used for visualizations of maps

## Common data pipeline target:

I, Evgeniy Ivlev, will use this file: `./datasets/cleaned/evaluation_fire_coordinates_date_feat_eng_2.csv` to produce my 1-13 month data and run my models.

![img_1.png](img_1.png)

This script produces the file above: `python ./datapipeline_panel_add_features.py`
## [Data pipeline diagram](https://docs.google.com/drawings/d/1JSGUZZg9EYoyRtfRQbYmxvmRRgAAAtKCh4ktoKaSbEA/edit)

![img.png](images/img.png)
### How to run the data pipeline:

You need to run these 3 files:

```commandline
python ./dataprep/evaluation_fonciere.py
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
python ./dataprep/main_evaluation_fonciere.py
```
You must run the `python ./dataprep/main_evaluation_fonciere.py` to get the file *evaluation_with_fire_and_coordinates_and_date.csv*
I did not commit it because it's 100MB big.

Note: alternatively, this command runs the whole pipeline
```commandline
python ./datapipeline_no_panel.py
```

# Panel data approach
This approach avoids issue with fire_date being null in the merged dataset when there is no fire. 
Each row is represented once per month to keep some homogeneity 
| ID_UEV        | Feature X      | MONTH | HAS_FIRE_IN_MONTH |
| ------------- | -------------- | ----- | ----------------- |
| 1234          | Value for 1234 |   1   |             False |
| 1234          | Value for 1234 |   2   |              True |
| 1234          | Value for 1234 |   ... |         ...       |
| 1234          | Value for 1234 |  12   |             False | 
| 4321          | Value for 4321 |   1   |             False |
| 4321          | Value for 4321 |  ...  |             False |
| 4321          | Value for 4321 |   12  |              True |

The HAS_FIRE_IN_MONTH column is True if there has been a fire within a 100m radius of this address in the specific MONTH.


## [Data pipeline diagram - Panel](https://docs.google.com/drawings/d/1LDBP_V14_hb_kPNOQbJvcjOaFdfQV9Tg8OQIGbXWYMY/edit?usp=sharing)
![panel_pipeline.png](images/panel_pipeline.png)
Note: items in red are yet to be completed

### [Running the panel data pipeline]
```commandline
python ./datapipeline_panel_month.py
```





# Option 2: (To be unified in second stage with the pipeline discussed above)

## [Data pipeline 2 diagram - Panel](https://docs.google.com/drawings/d/1tBfWPbFFkzylVUWRJzzGeF4eLe8oS1lH3CMAPt0VUFo/edit?usp=drive_link)
![panel_pipeline_2.png](images/panel_pipeline_2.png)


### [Running the panel data pipeline 2]
** One liner to run the whole pipeline
```commandline
python ./datapipeline_panel_add_features.py
```

This is equivalent to running the individual scripts below:
```commandline
# ====>eval_cleaned_feat_eng.csv
python ./dataprep/evaluation_fonciere.py

# ====> interventions_cleaned_with_has_fire.csv            
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py

# ====> eval_fire_Coordinates_date_feat_eng_2.csv
python ./dataprep/main_evaluation_feat_eng.py

# ====> building_month_fire_panel_feat_eng.csv
python ./dataprep/dense_panel_building_month.py
```
     
for time model run in dataprep:  time_model_Xgboost.ipynb   (When we run the same .py file we are having a memory error)

for forcasting and some visualizations run in dataprep:  time_model_Xgboost_forcasting_visualizatioons.ipynb 













