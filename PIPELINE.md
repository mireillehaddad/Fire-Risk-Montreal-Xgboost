# Fire Risk Montreal – End-to-End Pipeline

This document describes the complete data engineering and machine learning pipeline used in the Fire-Risk-Montreal-Xgboost project.

The pipeline transforms raw open datasets into a building-level monthly fire risk prediction model using XGBoost.

---

# 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install core packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost geopandas
pip install notebook nbconvert
pip install folium shapely
```



3. Step-by-Step Execution
Step 4 – Clean Evaluation Foncière
```bash
python .\dataprep\evaluation_fonciere.py

Input

datasets/raw/uniteevaluationfonciere.csv

Output

datasets/cleaned/eval_cleaned.csv

Step 5 – Process Fire Interventions
python .\dataprep\interventions_HAS_FIRE_binary_analysis.py

Input

datasets/raw/donneesouvertes-interventions-sim*.csv

Output

datasets/cleaned/fire_events.csv

Binary HAS_FIRE variable

Step 6 – Feature Engineering and Geospatial Merge
python .\dataprep\main_evaluation_feat_eng.py

Inputs

datasets/cleaned/eval_cleaned.csv

datasets/cleaned/adresses.csv

datasets/cleaned/fire_events.csv

Outputs

datasets/cleaned/evaluation_with_fire_and_coordinates.csv

datasets/merged/merged_evaluationfonciere_adresses.csv

Step 7 – Dense Monthly Panel Creation
python .\dataprep\dense_panel_building_month.py

Input

datasets/cleaned/evaluation_with_fire_and_coordinates.csv

Output

datasets/panel/panel_building_month.csv

dataprep/fire_risk_august_2025.csv (if generated)

Step 8 – XGBoost Model Training
python .\datamodel\xgboost_panel_with_feat.py

Input

datasets/panel/panel_building_month.csv

Outputs

models/xgb_fire_risk_model.json

outputs/predictions_monthly.csv

outputs/metrics_summary.csv

outputs/feature_importance.csv

Step 12 – Forecasting and Visualizations
python .\dataprep\time_model_Xgboost_forecasting_visualizations.py

Outputs

Time-based risk plots

Folium risk maps

Figures stored in docs/figures

Updated docs/index.html (if applicable)

Step 13 – Monthly Precision@K
python .\datamodel\monthly_precision_at_k.py

Output

outputs/monthly_precision_at_k.csv

Step 14 – Yearly Precision@K
python .\datamodel\yearly_precision_at_k_slide_numbers.py

Outputs

outputs/yearly_precision_at_k.csv

docs/slide_numbers.csv

4. Local Documentation Server
cd .\docs\
python -m http.server 8000

Open in browser:

http://localhost:8000
5. Public Deployment (GitHub Pages)

On GitHub:

Repository → Settings

Pages

Source: Deploy from a branch

Branch: main

Folder: /docs

Save

Public URL:

https://mireillehaddad.github.io/Fire-Risk-Montreal-Xgboost/
```