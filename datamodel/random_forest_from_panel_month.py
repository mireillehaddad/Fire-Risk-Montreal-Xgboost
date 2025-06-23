# Trains a simple random forest model
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'lib'))
import pandas as pd 
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

SOURCE_FILE = os.path.join('.','datasets','merged','base_panel_month_train.csv')
MODEL_OUTPUT = os.path.join('.','datamodel','base_panel_month_rf.pkl')
print("Loading training data ...")
df=pd.read_csv(SOURCE_FILE)
print("Training data loaded ...")

COLUMNS_TO_DROP=['ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE',
       'SUITE_DEBUT', 'LETTRE_DEBUT', 'LETTRE_FIN', 'CATEGORIE_UEF', 'MATRICULE83',
       'NOM_RUE_CLEAN', 'ADDR_DE','Unnamed: 0']

EXPECTED_COLUMNS=['MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT', 'ANNEE_CONSTRUCTION', 'CODE_UTILISATION', 'LIBELLE_UTILISATION', 'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT', 'NO_ARROND_ILE_CUM', 'LONGITUDE', 'LATITUDE', 'month', 'HAS_FIRE_IN_MONTH']



print("Trimming unnecessary features ...")
safe_drop_columns(df,COLUMNS_TO_DROP)
extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]

# Validate that all columns are accounted for, either dropped or kept
if len(extra_columns)>0:
    print("The following columns were not expected - please add them to either EXPECTED_COLUMNS or COLUMNS_TO_DROP and launch this script again")
    print(extra_columns)
    exit()


# Encode categorical columns
print("Encoding categorical columns ...")
categorical_cols = df.select_dtypes(include=['object']).columns

encoder_dict = {}
for col in categorical_cols:
    encoder_dict[col] = LabelEncoder()
    df[col] = encoder_dict[col].fit_transform(df[col])

# save label encoder for later
label_encoder_output = MODEL_OUTPUT.replace('.pkl','_label_enc.pkl')
with open(label_encoder_output, 'wb') as f:
    pickle.dump(encoder_dict, f)

# split into X,y
X=df.drop('HAS_FIRE_IN_MONTH',axis=1)
y=df['HAS_FIRE_IN_MONTH']


# %%
#
#define a random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training model ...")
rf.fit(X,y)

# Save the best model
print("Saving model...")
with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(rf, f)
print(f"Model saved to {MODEL_OUTPUT}")


