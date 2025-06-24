import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'lib'))
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle


from pathlib import Path

#Local import
from utils.date import print_timestamped_message


# 🔧 Project root directory (2 levels up from current script)
ROOT = Path(__file__).parents[1]
print(f"Root folder:{ROOT}")

# 🔹 Define input/output paths using ROOT
INPUT_CSV = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"
OUTPUT_CSV = ROOT / "datamodel" / "xgboost_panel_with_feat_label_enc_pred.csv"
ENCODER_FILE = ROOT / "datamodel" / "xgboost_panel_with_feat_label_enc.pkl"
MODEL_FILE = ROOT / "datamodel" / "xgboost_panel_with_feat.pkl"

# 🔍 Optional: check existence
print("[input exists?]", INPUT_CSV.exists(), "➜", INPUT_CSV)
#print("[output dir exists?]", OUTPUT_PANEL.parent.exists(), "➜", OUTPUT_PANEL.parent)


# 🔸 Load dataset
df = pd.read_csv(INPUT_CSV,parse_dates=['month'])


# # 🔸 Load panel for modeling
# df["month"] = pd.to_datetime(df["month"])
# df = df.sort_values(["ID_UEV", "month"])
# df["year"] = df["month"].dt.year


# 🔸 Add lag features
# for lag in range(1, 4):
#     df[f"fire_last_{lag}m"] = (
#         df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
#         .shift(lag)
#         .fillna(0)
#         .astype(int)
#     )

features = [
    "MUNICIPALITE", "ETAGE_HORS_SOL", "NOMBRE_LOGEMENT", "AGE_BATIMENT",
    "CODE_UTILISATION", "CATEGORIE_UEF", "SUPERFICIE_TERRAIN", "SUPERFICIE_BATIMENT",
    "NO_ARROND_ILE_CUM", "RATIO_SURFACE", "DENSITE_LOGEMENT", "HAS_MULTIPLE_LOGEMENTS",
    "FIRE_FREQUENCY_ZONE", "FIRE_RATE_ZONE", "FIRE_COUNT_LAST_YEAR_ZONE", "BUILDING_COUNT",
    "FIRE_RATE_ZONE_NORM", "FIRE_COUNT_LAST_YEAR_ZONE_NORM", 
    "fire_last_1m", "fire_last_2m", "fire_last_3m","fire_cumcount","fire_rolling_3m","fire_rolling_6m","fire_rolling_12m",
    "month_num", "year"  #,"season"
]
target = "HAS_FIRE_THIS_MONTH"




categories_encoders = {}
for col in ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]:
    categories_encoders[col]= LabelEncoder()
    df[col] = categories_encoders[col].fit_transform(df[col].astype(str))

# saving encoders to pickle file if we want to decode later, or encode new values
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(categories_encoders, f)

# %%
# First: convert in the full df
df["CATEGORIE_UEF"] = df["CATEGORIE_UEF"].astype("category")
df["NO_ARROND_ILE_CUM"] = df["NO_ARROND_ILE_CUM"].astype("category")

# Now re-split
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


print(train_df.columns.tolist())




print_timestamped_message("Training model ...")
# ⚖️ Class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    enable_categorical=True,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

print_timestamped_message("Saving model...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print_timestamped_message(f"Model saved to {MODEL_FILE}")

# 📊 Evaluate
print_timestamped_message("Evaluating model ...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# 🔍 Interpretation of Metrics (Threshold = 0.2?)
# Metric	Class 0 (No Fire)	Class 1 (Fire)
# Precision	0.992	0.027
# Recall	0.698	0.603
# F1-score	0.819	0.051
# 
# 🔵 High recall for fire class (1): You're catching 60% of fires, which is good for early detection.
# 
# 🔴 Very low precision for fire class (1): Among the predicted fires, only 2.7% are actually fires.
# 
# ⚖️ Accuracy is misleading (69%) due to imbalance (only 1.3% fires in your data).

# 🎯 Interpretation
# This is typical for imbalanced binary classification:
# 
# Your model is tuned toward catching more fires (high recall).
# 
# But it's imprecise: many "fire" predictions are wrong.
# 
# 

# %% [markdown]
# ✅ What You Did Well
# Feature engineering helped improve recall for rare event (fire).
# 
# Model is no longer "lazy" and defaulting to class 0.
# 
# Good step forward for early warning/risk flagging.
# 
# 



# tune the threshold and optimize the tradeoff between precision and recall using XGBoost and sklearn:

# %%
#✅ 1. Predict Probabilities on Test Set
# Predict probabilities (for class 1 = fire)
print_timestamped_message("Predicting test set probabilities")
y_probs = model.predict_proba(X_test)[:, 1]


result_test = X_test.copy(deep=True)
result_test['predicted_result']=y_pred
result_test['predicted_proba']=y_probs
result_test['target']=y_test
result_test.to_csv(OUTPUT_CSV,index=False)

