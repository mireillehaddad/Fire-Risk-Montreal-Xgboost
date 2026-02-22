# To run this code:
# python .\datamodel\xgboost_panel_with_feat.py

import sys
import os
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "lib"
    )
)

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

# Local import
#from utils.date import print_timestamped_message


ROOT = Path(__file__).parents[1]
print(f"Root folder: {ROOT}")

model_name = "xgboost_panel_with_feat"

INPUT_CSV = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"
OUTPUT_CSV = ROOT / "datamodel" / f"{model_name}_pred.csv"
ENCODER_FILE = ROOT / "datamodel" / f"{model_name}_label_enc.pkl"
FEATURE_LIST_FILE = ROOT / "datamodel" / f"{model_name}_features.pkl"
MODEL_FILE = ROOT / "datamodel" / f"{model_name}.pkl"

print("[input exists?]", INPUT_CSV.exists(), "->", INPUT_CSV)

print("Loading data ...")
df = pd.read_csv(INPUT_CSV, parse_dates=["month"])
print("Data loading complete.")


# -----------------------------
# Feature Selection
# -----------------------------
features = [
    "AGE_BATIMENT",
    "NO_ARROND_ILE_CUM",
    "RATIO_SURFACE",
    "DENSITE_LOGEMENT",
    "NOMBRE_LOGEMENT",
    "fire_cumcount",
    "fire_rolling_12m",
    "month_num",
    "year"
]

target = "HAS_FIRE_THIS_MONTH"


# -----------------------------
# Encode Categorical Variables
# -----------------------------
categories_encoders = {}

for col in ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]:
    categories_encoders[col] = LabelEncoder()
    df[col] = categories_encoders[col].fit_transform(df[col].astype(str))

with open(ENCODER_FILE, "wb") as f:
    pickle.dump(categories_encoders, f)

df["CATEGORIE_UEF"] = df["CATEGORIE_UEF"].astype("category")
df["NO_ARROND_ILE_CUM"] = df["NO_ARROND_ILE_CUM"].astype("category")


# -----------------------------
# Train / Test Split
# -----------------------------
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("Training features:", X_train.columns.tolist())


# -----------------------------
# Train Model
# -----------------------------
print("Training model ...")

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


# -----------------------------
# Save Model
# -----------------------------
print("Saving model...")

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_FILE}")

with open(FEATURE_LIST_FILE, "wb") as f:
    pickle.dump(features, f)

print(f"Feature list saved to {FEATURE_LIST_FILE}")


# -----------------------------
# Evaluate
# -----------------------------
print("Evaluating model ...")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# -----------------------------
# Save Predictions
# -----------------------------
print("Predicting test set probabilities")

y_probs = model.predict_proba(X_test)[:, 1]

result_test = test_df.copy(deep=True)
result_test["predicted_result"] = y_pred
result_test["predicted_proba"] = y_probs
result_test["target"] = y_test

result_test.to_csv(OUTPUT_CSV, index=False)

print(f"Saved test set predictions to {OUTPUT_CSV}")