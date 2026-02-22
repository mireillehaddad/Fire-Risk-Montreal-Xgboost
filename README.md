Inside Fire-Risk-Montreal:

# 1.  Create venv
python -m venv .venv
# 2.  Activate
.\.venv\Scripts\Activate
# 3. Install core packages
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
pip install geopandas

Test-Path .\datasets\raw\uniteevaluationfonciere.csv
Test-Path .\datasets\raw\donneesouvertes-interventions-sim.csv
Test-Path .\datasets\raw\donneesouvertes-interventions-sim2020.csv
Test-Path .\datasets\cleaned\adresses.csv
# 4. Run python .\dataprep\evaluation_fonciere.py
# 5. Run python .\dataprep\interventions_HAS_FIRE_binary_analysis.py
# 6. Run python .\dataprep\main_evaluation_feat_eng.py
# 7. Run python .\dataprep\dense_panel_building_month.py
# 8. Run python .\datamodel\xgboost_panel_with_feat.py

# 9. pip install notebook nbconvert 
# 10. Run python -m nbconvert --to script dataprep/time_model_Xgboost_forecasting_visualizations.ipynb
# 11. pip install folium geopandas shapely


# 12. Run python  .\dataprep\time_model_Xgboost_forecasting_visualizations.py
# 13. Run python .\datamodel\monthly_precision_at_k.py
# 14. Run python .\datamodel\yearly_precision_at_k_slide_numbers.py

# 15. cd .\docs\    
# 16. Run python -m http.server 8000  (This is only locally)

# To see publically
Enable GitHub Pages

On GitHub:

Repo → Settings

Pages

Source: Deploy from a branch

Branch: main

Folder: /docs

Save

Then your public URL will look like:
https://mireillehaddad.github.io/Fire-Risk-Montreal-Xgboost/

And your homepage:
https://mireillehaddad.github.io/Fire-Risk-Montreal-Xgboost/index.html

