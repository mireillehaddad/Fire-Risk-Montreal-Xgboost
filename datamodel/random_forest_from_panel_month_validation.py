# Validates the model saved via pickle against 2024-2025 data
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'lib'))

import pandas as pd 
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,make_scorer,precision_score
from sklearn.preprocessing import LabelEncoder

#Local import
from utils.date import extract_date_components
from utils.generic_df import safe_drop_columns

SOURCE_FILE = os.path.join('.','datasets','merged','base_panel_month_test.csv')
MODEL_OUTPUT = os.path.join('.','datamodel','base_panel_month_rf.pkl')
PROBA_OUTPUT = os.path.join('.','datamodel','random_forest_from_panel_month_proba.csv')
print("Loading test data ...")
df=pd.read_csv(SOURCE_FILE)

print("dropping columns")
COLUMNS_TO_DROP=['ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE',
       'SUITE_DEBUT', 'LETTRE_DEBUT', 'LETTRE_FIN', 'CATEGORIE_UEF', 'MATRICULE83',
       'NOM_RUE_CLEAN', 'ADDR_DE','Unnamed: 0']

EXPECTED_COLUMNS=['MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT', 'ANNEE_CONSTRUCTION', 'CODE_UTILISATION', 'LIBELLE_UTILISATION', 'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT', 'NO_ARROND_ILE_CUM', 'LONGITUDE', 'LATITUDE', 'month', 'HAS_FIRE_IN_MONTH']

safe_drop_columns(df,COLUMNS_TO_DROP)
extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]

# Validate that all columns are accounted for, either dropped or kept
if len(extra_columns)>0:
    print("The following columns were not expected - please add them to either EXPECTED_COLUMNS or COLUMNS_TO_DROP and launch this script again")
    print(extra_columns)
    exit()


# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
# load saved label encoder
label_encoder_output = MODEL_OUTPUT.replace('.pkl','_label_enc.pkl')
with open(label_encoder_output, 'rb') as f:
    encoder_dict = pickle.load(f)
for col in categorical_cols:
    df[col] = encoder_dict[col].transform(df[col])

# split into X,y
X_test=df.drop('HAS_FIRE_IN_MONTH',axis=1)
y_test=df['HAS_FIRE_IN_MONTH']

# Verify the save worked
with open(MODEL_OUTPUT, 'rb') as f:
    loaded_model = pickle.load(f)

# validate model performance on test set
print("predicting risk from saved model...")
y_test_predictions = loaded_model.predict(X_test)
print("calculating prediction metrics")
test_accuracy = accuracy_score(y_test,y_test_predictions)
test_precision = precision_score(y_test,y_test_predictions)
test_recall = recall_score(y_test,y_test_predictions)
test_f1 = f1_score(y_test,y_test_predictions)
print(f"\nTest Set Accuracy: {test_accuracy:.2%}")
print(f"\nTest Set Precision: {test_precision:.2%}")
print(f"\nTest Set Recall: {test_recall:.2%}")
print(f"\nTest Set F1-score: {test_f1:.2%}")

print("Saving proba for display")
y_test_predict_proba = loaded_model.predict_proba(X_test)[:, 1]

X_test['predicted_fire']=y_test_predictions
X_test['predicted_proba']=y_test_predict_proba
X_test['HAS_FIRE_IN_MONTH']=y_test
X_test.to_csv(PROBA_OUTPUT,index=False)

print("Random pick: ")
print(X_test.iloc[np.random.choice(X_test.index.to_list(),size=7500)]['HAS_FIRE_IN_MONTH'].value_counts())
print("Predicted :")
print(X_test.nlargest(7500,columns='predicted_proba')['HAS_FIRE_IN_MONTH'].value_counts())

exit()
print("Determining feature importance ...")

loaded_model.feature_importances_

# Built-in feature importance (Gini Importance)
feature_names=list(X_test.columns)
importances = loaded_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
print(feature_imp_df)


# 


