```mermaid
fix this # Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install core packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost geopandas
pip install notebook nbconvert
pip install folium shapely


# 2. Data Pipeline Overview
```mermaid
flowchart TD

  A1["datasets/raw/uniteevaluationfonciere.csv"]
  A2["datasets/raw/donneesouvertes-interventions-sim.csv"]
  A3["datasets/raw/donneesouvertes-interventions-sim2020.csv"]
  A4["datasets/cleaned/adresses.csv"]

  S4["dataprep/evaluation_fonciere.py"]
  O4["datasets/cleaned/eval_cleaned.csv"]

  S5["dataprep/interventions_HAS_FIRE_binary_analysis.py"]
  O5["datasets/cleaned/fire_events.csv"]

  S6["dataprep/main_evaluation_feat_eng.py"]
  O6["datasets/cleaned/evaluation_with_fire_and_coordinates.csv"]

  S7["dataprep/dense_panel_building_month.py"]
  O7["datasets/panel/panel_building_month.csv"]

  S8["datamodel/xgboost_panel_with_feat.py"]
  O8a["models/xgb_fire_risk_model.json"]
  O8b["outputs/predictions_monthly.csv"]

  S12["dataprep/time_model_Xgboost_forecasting_visualizations.py"]
  O12["docs/figures/*.png"]

  S13["datamodel/monthly_precision_at_k.py"]
  S14["datamodel/yearly_precision_at_k_slide_numbers.py"]

  A1 --> S4 --> O4
  A2 --> S5 --> O5
  A3 --> S5
  O4 --> S6
  A4 --> S6
  O5 --> S6 --> O6
  O6 --> S7 --> O7
  O7 --> S8 --> O8a
  O7 --> S8 --> O8b
  O8b --> S13
  O8b --> S14
  O8b --> S12 --> O12\

  ```



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
```
Input

datasets/raw/uniteevaluationfonciere.csv

Output

datasets/cleaned/eval_cleaned.csv

Step 5 – Process Fire Interventions
```bash
python .\dataprep\interventions_HAS_FIRE_binary_analysis.py
```
Input

datasets/raw/donneesouvertes-interventions-sim*.csv

Output

datasets/cleaned/fire_events.csv

Binary HAS_FIRE variable

Step 6 – Feature Engineering and Geospatial Merge
```bash
python .\dataprep\main_evaluation_feat_eng.py
```
Inputs

datasets/cleaned/eval_cleaned.csv

datasets/cleaned/adresses.csv

datasets/cleaned/fire_events.csv

Outputs

datasets/cleaned/evaluation_with_fire_and_coordinates.csv

datasets/merged/merged_evaluationfonciere_adresses.csv

Step 7 – Dense Monthly Panel Creation
```bash 
python .\dataprep\dense_panel_building_month.py
```
Input

datasets/cleaned/evaluation_with_fire_and_coordinates.csv

Output

datasets/panel/panel_building_month.csv

dataprep/fire_risk_august_2025.csv (if generated)

Step 8 – XGBoost Model Training
```bash 
python .\datamodel\xgboost_panel_with_feat.py
```
Input

datasets/panel/panel_building_month.csv

Outputs

models/xgb_fire_risk_model.json

outputs/predictions_monthly.csv

outputs/metrics_summary.csv

outputs/feature_importance.csv

Step 12 – Forecasting and Visualizations
``` bash 
python .\dataprep\time_model_Xgboost_forecasting_visualizations.py
```
Outputs

Time-based risk plots

Folium risk maps

Figures stored in docs/figures

Updated docs/index.html (if applicable)

Step 13 – Monthly Precision@K
```bash
python .\datamodel\monthly_precision_at_k.py
```
Output

outputs/monthly_precision_at_k.csv

Step 14 – Yearly Precision@K
```bash
python .\datamodel\yearly_precision_at_k_slide_numbers.py
```
Outputs

outputs/yearly_precision_at_k.csv

docs/slide_numbers.csv

4. Local Documentation Server
```bash
cd .\docs\
python -m http.server 8000
```
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
