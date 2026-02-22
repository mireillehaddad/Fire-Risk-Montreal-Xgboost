# To Run this code:  python .\dataprep\evaluation_fonciere.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ORIGINAL_FILE_NAME_EVAL = "./datasets/raw/uniteevaluationfonciere.csv"
DESTINATION_FILE_NAME = "./datasets/cleaned/eval_cleaned.csv"
DESTINATION_FILE_NAME_FEAT_ENG = "./datasets/cleaned/eval_cleaned_feat_eng.csv"

df_eval = pd.read_csv(ORIGINAL_FILE_NAME_EVAL)

df_eval.info()

# Clean ANNEE_CONSTRUCTION
mask = (df_eval["ANNEE_CONSTRUCTION"] < 1800) | (df_eval["ANNEE_CONSTRUCTION"] > 2025)
df_eval.loc[mask, "ANNEE_CONSTRUCTION"] = np.nan

total_invalid = mask.sum()
print(
    f"Marked {total_invalid} entries ({total_invalid/len(df_eval)*100:.2f}%) as missing for ANNEE_CONSTRUCTION"
)

df_eval["ANNEE_CONSTRUCTION"] = df_eval["ANNEE_CONSTRUCTION"].fillna(0)

df_eval["_ANNEE_CONSTRUCTION_NUM"] = pd.to_numeric(
    df_eval["ANNEE_CONSTRUCTION"], errors="coerce"
)

mask_missing = df_eval["_ANNEE_CONSTRUCTION_NUM"].isna()
mask_buildings = df_eval["LIBELLE_UTILISATION"].str.contains(
    "Logement|Immeuble", case=False, na=False
)
mask_to_impute = mask_missing & mask_buildings

median_years_by_borough = (
    df_eval.loc[~mask_missing]
    .groupby("NO_ARROND_ILE_CUM")["_ANNEE_CONSTRUCTION_NUM"]
    .median()
    .astype("Int64")
)

for borough, median_year in median_years_by_borough.items():
    idx = mask_to_impute & (df_eval["NO_ARROND_ILE_CUM"] == borough)
    df_eval.loc[idx, "_ANNEE_CONSTRUCTION_NUM"] = median_year

df_eval["ANNEE_CONSTRUCTION"] = (
    df_eval["_ANNEE_CONSTRUCTION_NUM"].fillna("unknown").astype(str)
)
df_eval.drop(columns=["_ANNEE_CONSTRUCTION_NUM"], inplace=True)

print("ANNEE_CONSTRUCTION cleaning complete.")

# Missing summary
columns_to_check = ["NOMBRE_LOGEMENT", "ETAGE_HORS_SOL", "SUPERFICIE_BATIMENT"]
missing_summary = df_eval[columns_to_check].isna().sum().to_frame(name="Missing Count")
missing_summary["Missing %"] = 100 * missing_summary["Missing Count"] / len(df_eval)
print(missing_summary)

# Impute NOMBRE_LOGEMENT
median_units = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["NOMBRE_LOGEMENT"]
    .median()
    .dropna()
)

def impute_nombre_logement(row):
    if pd.isna(row["NOMBRE_LOGEMENT"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_units.get(key, np.nan)
    return row["NOMBRE_LOGEMENT"]

df_eval["NOMBRE_LOGEMENT"] = df_eval.apply(impute_nombre_logement, axis=1)

borough_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["NOMBRE_LOGEMENT"].median()

def fallback_impute_logement(row):
    if pd.isna(row["NOMBRE_LOGEMENT"]):
        return borough_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["NOMBRE_LOGEMENT"]

df_eval["NOMBRE_LOGEMENT"] = df_eval.apply(fallback_impute_logement, axis=1)

print("Final missing NOMBRE_LOGEMENT:", df_eval["NOMBRE_LOGEMENT"].isna().sum())

# Impute ETAGE_HORS_SOL
median_etages = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["ETAGE_HORS_SOL"]
    .median()
    .dropna()
)

def impute_etage(row):
    if pd.isna(row["ETAGE_HORS_SOL"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_etages.get(key, np.nan)
    return row["ETAGE_HORS_SOL"]

df_eval["ETAGE_HORS_SOL"] = df_eval.apply(impute_etage, axis=1)

borough_etage_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["ETAGE_HORS_SOL"].median()

def fallback_impute_etage(row):
    if pd.isna(row["ETAGE_HORS_SOL"]):
        return borough_etage_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["ETAGE_HORS_SOL"]

df_eval["ETAGE_HORS_SOL"] = df_eval.apply(fallback_impute_etage, axis=1)

print("Final missing ETAGE_HORS_SOL:", df_eval["ETAGE_HORS_SOL"].isna().sum())

# Impute SUPERFICIE_BATIMENT
median_surface = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["SUPERFICIE_BATIMENT"]
    .median()
    .dropna()
)

def impute_surface(row):
    if pd.isna(row["SUPERFICIE_BATIMENT"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_surface.get(key, np.nan)
    return row["SUPERFICIE_BATIMENT"]

df_eval["SUPERFICIE_BATIMENT"] = df_eval.apply(impute_surface, axis=1)

borough_surface_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["SUPERFICIE_BATIMENT"].median()

def fallback_impute_surface(row):
    if pd.isna(row["SUPERFICIE_BATIMENT"]):
        return borough_surface_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["SUPERFICIE_BATIMENT"]

df_eval["SUPERFICIE_BATIMENT"] = df_eval.apply(fallback_impute_surface, axis=1)

print("Final missing SUPERFICIE_BATIMENT:", df_eval["SUPERFICIE_BATIMENT"].isna().sum())

# Save cleaned dataset
df_eval.to_csv(DESTINATION_FILE_NAME, index=False)

# Feature Engineering
df = pd.read_csv(DESTINATION_FILE_NAME)
current_year = 2025

df["AGE_BATIMENT"] = df["ANNEE_CONSTRUCTION"].apply(
    lambda x: current_year - int(float(x)) if x != "unknown" else np.nan
)

df["RATIO_SURFACE"] = df["SUPERFICIE_BATIMENT"] / df["SUPERFICIE_TERRAIN"]
df["RATIO_SURFACE"] = df["RATIO_SURFACE"].replace([np.inf, -np.inf], np.nan)

df["DENSITE_LOGEMENT"] = df["NOMBRE_LOGEMENT"] / df["SUPERFICIE_BATIMENT"]
df["DENSITE_LOGEMENT"] = df["DENSITE_LOGEMENT"].replace([np.inf, -np.inf], np.nan)

df["HAS_MULTIPLE_LOGEMENTS"] = (df["NOMBRE_LOGEMENT"] > 1).astype(int)

df["FIRE_FREQUENCY_ZONE"] = df["NO_ARROND_ILE_CUM"].map(
    df["NO_ARROND_ILE_CUM"].value_counts()
)

# Normalize safely
scaler = MinMaxScaler()
to_normalize = ["AGE_BATIMENT", "RATIO_SURFACE", "DENSITE_LOGEMENT", "FIRE_FREQUENCY_ZONE"]

X = df[to_normalize].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.clip(lower=-1e12, upper=1e12)

df[to_normalize] = scaler.fit_transform(X)

# Final dataset
original_cols = [
    "ID_UEV", "CIVIQUE_DEBUT", "CIVIQUE_FIN", "NOM_RUE", "SUITE_DEBUT",
    "MUNICIPALITE", "ETAGE_HORS_SOL", "NOMBRE_LOGEMENT",
    "ANNEE_CONSTRUCTION", "CODE_UTILISATION", "LETTRE_DEBUT",
    "LETTRE_FIN", "LIBELLE_UTILISATION", "CATEGORIE_UEF",
    "MATRICULE83", "SUPERFICIE_TERRAIN",
    "SUPERFICIE_BATIMENT", "NO_ARROND_ILE_CUM"
]

engineered_cols = [
    "AGE_BATIMENT", "RATIO_SURFACE",
    "DENSITE_LOGEMENT", "HAS_MULTIPLE_LOGEMENTS",
    "FIRE_FREQUENCY_ZONE"
]

df_final = df[original_cols + engineered_cols]

df_final.to_csv(DESTINATION_FILE_NAME_FEAT_ENG, index=False)

print("Feature engineering complete.")