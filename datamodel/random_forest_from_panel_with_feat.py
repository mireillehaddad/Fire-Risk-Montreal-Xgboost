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
from sklearn.metrics import accuracy_score,recall_score,f1_score,make_scorer,precision_score,classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE



#Local import
from utils.date import extract_date_components,print_timestamped_message
from utils.generic_df import safe_drop_columns

SOURCE_FILE = os.path.join('.','datasets','cleaned','building_month_fire_panel_feat_eng.csv')
MODEL_OUTPUT = os.path.join('.','datamodel','rf_panel_with_feat.pkl')
OUTPUT_CSV = os.path.join('.','datamodel','rf_panel_with_feat_results.csv')
print_timestamped_message("Loading training data ...")
df=pd.read_csv(SOURCE_FILE)
print_timestamped_message("Training data loaded ...")

COLUMNS_TO_DROP=['ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE',
       'SUITE_DEBUT', 'LETTRE_DEBUT', 'LETTRE_FIN', 'CATEGORIE_UEF', 'MATRICULE83',
       'NOM_RUE_CLEAN', 'ADDR_DE','Unnamed: 0']

features=[
    "MUNICIPALITE", "ETAGE_HORS_SOL", "NOMBRE_LOGEMENT", "AGE_BATIMENT",
    "CODE_UTILISATION", "CATEGORIE_UEF", "SUPERFICIE_TERRAIN", "SUPERFICIE_BATIMENT",
    "NO_ARROND_ILE_CUM", #"RATIO_SURFACE", "DENSITE_LOGEMENT",
      "HAS_MULTIPLE_LOGEMENTS",
    "FIRE_FREQUENCY_ZONE", "FIRE_RATE_ZONE", "FIRE_COUNT_LAST_YEAR_ZONE", "BUILDING_COUNT",
    "FIRE_RATE_ZONE_NORM", "FIRE_COUNT_LAST_YEAR_ZONE_NORM", 
    "fire_last_1m", "fire_last_2m", "fire_last_3m","fire_cumcount","fire_rolling_3m","fire_rolling_6m","fire_rolling_12m",
    "month_num", "year"  #,"season"
]
target = 'HAS_FIRE_THIS_MONTH'

# Encode categorical columns
print_timestamped_message("Encoding categorical columns ...")
#categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]
encoder_dict = {}
for col in categorical_cols:
    encoder_dict[col] = LabelEncoder()
    df[col] = encoder_dict[col].fit_transform(df[col])

# save label encoder for later
label_encoder_output = MODEL_OUTPUT.replace('.pkl','_label_enc.pkl')
with open(label_encoder_output, 'wb') as f:
    pickle.dump(encoder_dict, f)

# split into train,test and X,y
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]
print(f"missing values {X_test.isnull().sum()}")
print(X_test.describe())
print(X_test.info())
print(X_test.isna().any())
print(f"X_train {X_train.shape}")
print(f"X_test {X_test.shape}")
print(f"y_train {y_train.shape}")
print(f"y_test {y_test.shape}")

#Use SMOTE to oversample the minority class
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
#using balanced random forest to compensate for class imbalance
#rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
print(f"over X : {over_X.isnull().sum()}")
print_timestamped_message("Training model ...")
#rf.fit(X_train,y_train)
rf.fit(over_X,over_y)

# Save the best model
print_timestamped_message("Saving model...")
with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(rf, f)
print(f"Model saved to {MODEL_OUTPUT}")

print("Validating model ...")
print_timestamped_message("Predicting target values for X_test")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

print_timestamped_message("Predicting probabilities")
y_proba = rf.predict_proba(X_test)[:,1]
print_timestamped_message("Saving results")
result_test = X_test.copy(deep=True)
result_test['predicted_result']=y_pred
result_test['predicted_proba']=y_proba
result_test['target']=y_test
result_test.to_csv(OUTPUT_CSV,index=False)
print_timestamped_message(f"End of {__file__}")


# #2025-06-24 13:31:08 Predicting target values for X_test
#               precision    recall  f1-score   support

#            0      0.988     0.931     0.958   3674405
#            1      0.027     0.139     0.045     50239

#     accuracy                          0.920   3724644
#    macro avg      0.507     0.535     0.502   3724644
# weighted avg      0.975     0.920     0.946   3724644
