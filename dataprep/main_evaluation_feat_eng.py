#To run  this code python ./dataprep/main_evaluation_feat_eng.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import MinMaxScaler


# -------------------------------
# 📁 Load Data
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]


EVAL_PATH = ROOT / "datasets" / "cleaned" / "eval_cleaned_feat_eng.csv"
ADDR_PATH = ROOT / "datasets" / "cleaned" / "adresses.csv"
INC_PATH  = ROOT / "datasets" / "cleaned" / "interventions_cleaned_with_has_fire.csv"

print("[eval exists?]", EVAL_PATH.exists(), "\t➜", EVAL_PATH)
print("[addr exists?]", ADDR_PATH.exists(), "\t➜", ADDR_PATH)
print("[inc exists?] ", INC_PATH.exists(),  "\t➜", INC_PATH)




# Load CSVs
if not EVAL_PATH.exists():
    raise FileNotFoundError(f"❌ EVAL file not found at {EVAL_PATH}")


# -------------------------------
# 🧹 Preprocessing
# -------------------------------
eval_df = pd.read_csv(EVAL_PATH, dtype=str)
addr_df = pd.read_csv(ADDR_PATH, dtype=str)
inc_df  = pd.read_csv(INC_PATH)



# Clean evaluation
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\s+\(.*)?$")[0].str.lower().str.strip()
original_eval_df = eval_df.copy()


# Clean addresses
addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = (
    addr_df["GENERIQUE"].str.lower().str.strip() + " " +
    addr_df["SPECIFIQUE"].str.lower().str.strip()
)


# Merge to get coordinates
eval_with_coords = pd.merge(eval_df, addr_df,
                            left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                            right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                            how="left")
eval_with_coords = eval_with_coords.dropna(subset=["LONGITUDE", "LATITUDE"])

# Convert to GeoDataFrame
eval_gdf = gpd.GeoDataFrame(
    eval_with_coords,
    geometry=gpd.points_from_xy(eval_with_coords["LONGITUDE"].astype(float),
                                 eval_with_coords["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)




# Clean & filter fire incidents
inc_df = inc_df[inc_df["DESCRIPTION_GROUPE"].str.contains("INCENDIE", case=False, na=False)]
inc_df["CREATION_DATE_TIME"] = pd.to_datetime(inc_df["CREATION_DATE_TIME"], errors='coerce')
incident_gdf = gpd.GeoDataFrame(
    inc_df,
    geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
    crs="EPSG:4326"
)



# Project to meters for spatial join
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = incident_gdf.to_crs(epsg=32188)
incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")




# Spatial join to assign fires to properties
joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')
joined = joined.rename(columns={"CREATION_DATE_TIME": "fire_date"})
joined["fire"] = True



# Drop irrelevant columns
drop_cols = [
    "CIVIQUE_DEBUT", "CIVIQUE_FIN", "NOM_RUE", "LETTRE_DEBUT", "LETTRE_FIN",
    "MATRICULE83", "NOM_RUE_CLEAN", "ADDR_DE", "X", "Y", "geometry",
    "geometry_right", "index_right", "DESCRIPTION_GROUPE", "INCIDENT_TYPE_DESC",
    "DIVISION", "NOM_VILLE", "NOM_ARROND"
]
joined.drop(columns=drop_cols, inplace=True, errors="ignore")


# Extract relevant fire information
fire_records = joined[["ID_UEV", "fire_date", "NOMBRE_UNITES", "CASERNE"]].copy()
fire_records["fire"] = True



# Merge fire info back into original evaluation
data = pd.merge(original_eval_df, fire_records, on="ID_UEV", how="left")
data["fire"] = data["fire"].fillna(False)
data["fire_date"] = pd.to_datetime(data["fire_date"], errors="coerce")



# Reattach coordinates
addr_coords = addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]]
data = pd.merge(data,
                addr_coords,
                left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                how="left")



# --------------------------------------
# ⏳ Time-Related Feature Engineering
# --------------------------------------
data["fire_month"] = data["fire_date"].dt.month
data["fire_year"] = data["fire_date"].dt.year




def get_season(month):
    if pd.isnull(month): return None
    if month in [12, 1, 2]: return "Winter"
    if month in [3, 4, 5]: return "Spring"
    if month in [6, 7, 8]: return "Summer"
    return "Fall"



data["fire_season"] = data["fire_month"].apply(get_season)
data["year_month"] = data["fire_date"].dt.to_period("M").astype(str)




# Save enriched dataset
#data.to_csv("evaluation_fire_coordinates_date_feat_eng.csv", index=False)



# --------------------------------------
# 📊 FIRE_COUNT_LAST_YEAR_ZONE + RATE
# --------------------------------------
data["NO_ARROND_ILE_CUM"] = data["NO_ARROND_ILE_CUM"].astype(str)
fires_2024 = data[(data["fire"] == True) & (data["fire_date"].dt.year == 2024)].copy()
fire_count = fires_2024.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="FIRE_COUNT_LAST_YEAR_ZONE")
building_count = data.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="BUILDING_COUNT")



# Merge zone stats
data = data.merge(fire_count, on="NO_ARROND_ILE_CUM", how="left")
data = data.merge(building_count, on="NO_ARROND_ILE_CUM", how="left")
data["FIRE_COUNT_LAST_YEAR_ZONE"] = data["FIRE_COUNT_LAST_YEAR_ZONE"].fillna(0)
data["FIRE_RATE_ZONE"] = (data["FIRE_COUNT_LAST_YEAR_ZONE"] / data["BUILDING_COUNT"]).fillna(0)




# Normalize
scaler = MinMaxScaler()
data[["FIRE_COUNT_LAST_YEAR_ZONE_NORM", "FIRE_RATE_ZONE_NORM"]] = scaler.fit_transform(
    data[["FIRE_COUNT_LAST_YEAR_ZONE", "FIRE_RATE_ZONE"]]
)




# Save version
#data.to_csv("eval_fire_coordinates_date_feat_eng_1.csv", index=False)
#print("✅  file saved: eval_fire_coordinates_date_feat_eng_1.csv")



data.head()




# 🔍 OPTIONAL CLEANUP & VALIDATION ------------------------------

# 1.  missing coordinates (optional — depends on modeling strategy)
num_missing_coords = data[["LATITUDE", "LONGITUDE"]].isna().any(axis=1).sum()
print(f"⚠️ Rows with missing coordinates: {num_missing_coords}")
# Uncomment below if you want to drop them:
# data = data.dropna(subset=["LATITUDE", "LONGITUDE"])

# 2. Binary indicator for fire occurrence
data["had_fire"] = data["fire_date"].notna().astype(int)

# 3. Final sanity checks
print("\n✅ Final Features Summary:")
print(f"Total rows           : {len(data)}")
print(f"Rows with fire       : {data['fire'].sum()}")
print(f"Rows without fire    : {(~data['fire']).sum()}")
print(f"Rows with fire date  : {data['fire_date'].notna().sum()}")
print(f"Rows with 'had_fire' : {data['had_fire'].sum()}")
print(f"Columns available    : {len(data.columns)}")
print("\n🧾 Feature columns:\n", sorted(data.columns.tolist()))


# ✅ Your Current Dataset Snapshot
# Total records: 663,783
# 
# Rows with valid coordinates: ≈ 461,882
# 
# Fire cases (binary): 294,767 (fire and had_fire both indicate this)
# 
# Missing coordinates: 201,901 rows (⚠️ optional to drop depending on modeling use case)
# 
# Total features: 41



# 🧠 Engineered Features Present
# You've already added:
# 
# Building-related: AGE_BATIMENT, DENSITE_LOGEMENT, RATIO_SURFACE, HAS_MULTIPLE_LOGEMENTS
# 
# Fire statistics: FIRE_COUNT_LAST_YEAR_ZONE, FIRE_RATE_ZONE, and their normalized versions
# 
# Time features: fire_month, fire_year, fire_season, year_month, had_fire
# 
# Spatial context: NO_ARROND_ILE_CUM, FIRE_FREQUENCY_ZONE, LATITUDE, LONGITUDE

# Mark missing coordinates
# 
# Analyze if fire occurrence is associated with missingness
# 
# Summarize useful group comparisons
# 
# Optionally keep a flag for modeling



# 1️⃣ Mark rows with missing coordinates
data["missing_coords"] = data[["LATITUDE", "LONGITUDE"]].isna().any(axis=1)



# 2️⃣ Compare fire rates for rows with vs without coordinates
fire_by_coords = data.groupby("missing_coords")["fire"].value_counts(normalize=True).unstack().fillna(0)
print("🔥 Fire distribution by coordinate presence:")
print(fire_by_coords)


# Null fire dates for non-fire buildings.
# → Solved by generating a full panel (every building × month).

# 🔁 Why We Get Null fire_date
# When you merge fire incident data (with fire_date) onto your building dataset, only the buildings that had a matching fire incident get a fire_date.
# 
# So:
# 
# Buildings with a fire → fire_date = valid date
# 
# Buildings without a fire → fire_date = NaT (null)
# 
# ✅ How to Handle This Properly for Time-Series Modeling
# The best practice is to build a complete panel dataset:
# 
# ID_UEV	year_month	fire
# 123456	2023-01	0
# 123456	2023-02	0
# 123456	2023-03	1
# 123456	2023-04	0
# ...	...	...
# 
# This way:
# 
# Every building appears once per month, even if no fire occurred.
# 
# fire = 1 where a fire occurred.
# 
# fire = 0 where it didn’t.
# 
# fire_date only exists where fire = 1.
# 
# 🧱 Why This Solves the fire_date = NaT Issue
# It’s not a bug, it’s an expected outcome.
# 
# You’re no longer trying to “impute” or “guess” fire dates.
# 
# The fire column is the true label for your classifier.
# 
# fire_date becomes optional metadata, not a required feature.
# 
# 
# 
# 

# # Features selection

# ✅ Core Structural Features (building characteristics)
# | Feature                                     | Keep?          | Reason                                                     |
# | ------------------------------------------- | -------------- | ---------------------------------------------------------- |
# | `ID_UEV`                                    | ✔️ (reference) | Unique identifier – not a feature, but useful for tracking |
# | `CIVIQUE_DEBUT`, `CIVIQUE_FIN`              | ❌              | Redundant with coordinates and `ADDR_DE`                   |
# | `NOM_RUE`, `NOM_RUE_CLEAN`, `ADDR_DE`       | ❌              | Redundant; if coordinates used                             |
# | `SUITE_DEBUT`                               | ❌              | Often sparse and not predictive                            |
# | `MUNICIPALITE`                              | ✔️             | Categorical location info                                  |
# | `ETAGE_HORS_SOL`                            | ✔️             | Structural height – numeric                                |
# | `NOMBRE_LOGEMENT`                           | ✔️             | Residential density                                        |
# | `ANNEE_CONSTRUCTION`, `AGE_BATIMENT`        | ✔️             | Use `AGE_BATIMENT` – drop raw year in model                |
# | `CODE_UTILISATION`, `CATEGORIE_UEF`         | ✔️             | Building usage (categorical)                               |
# | `SUPERFICIE_TERRAIN`, `SUPERFICIE_BATIMENT` | ✔️             | Important physical dimensions                              |
# | `RATIO_SURFACE`, `DENSITE_LOGEMENT`         | ✔️             | Engineered structural density                              |
# | `HAS_MULTIPLE_LOGEMENTS`                    | ✔️             | Boolean – keep                                             |
# | `MATRICULE83`                               | ❌              | Internal ID – drop                                         |
# 🧭 Spatial Features
# 
# | Feature                 | Keep? | Reason                                     |
# | ----------------------- | ----- | ------------------------------------------ |
# | `NO_ARROND_ILE_CUM`     | ✔️    | Use as location index if aggregating       |
# | `LONGITUDE`, `LATITUDE` | ✔️    | Needed for spatial models                  |
# | `missing_coords`        | ✔️    | May be used to flag imputed/missing values |
# | `FIRE_FREQUENCY_ZONE`   | ✔️    | Feature engineering – keep                 |
# 
# 🔥 Target & Fire Incident Features
# | Feature         | Keep? | Reason                                                      |
# | --------------- | ----- | ----------------------------------------------------------- |
# | `fire`          | ✔️    | Primary binary target                                       |
# | `fire_date`     | ✔️    | Needed for panel & temporal features                        |
# | `had_fire`      | ✔️    | Useful for binary summary features                          |
# | `NOMBRE_UNITES` | ✔️    | Fire response intensity – could correlate with severity     |
# | `CASERNE`       | ❌     | Operational info, likely redundant and possibly a data leak |
# 🕒 Time Features (from fire_date) 
# | Feature                                  | Keep? | Reason                          |
# | ---------------------------------------- | ----- | ------------------------------- |
# | `fire_month`, `fire_year`, `fire_season` | ✔️    | Categorical time-based features |
# | `year_month`                             | ✔️    | Needed for panel data alignment |
# 🔁 Zone-Level Aggregates
# | Feature                                                 | Keep? | Reason                                |
# | ------------------------------------------------------- | ----- | ------------------------------------- |
# | `FIRE_COUNT_LAST_YEAR_ZONE`                             | ✔️    | Engineered from past data             |
# | `BUILDING_COUNT`                                        | ✔️    | Needed for calculating rates          |
# | `FIRE_RATE_ZONE`                                        | ✔️    | Strong predictor                      |
# | `FIRE_COUNT_LAST_YEAR_ZONE_NORM`, `FIRE_RATE_ZONE_NORM` | ✔️    | Normalized versions – may help models |
#   
#   
#   Keep these columns:
# [
#  'ID_UEV', 'MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT',
#  'AGE_BATIMENT', 'CODE_UTILISATION', 'CATEGORIE_UEF',
#  'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT',
#  'RATIO_SURFACE', 'DENSITE_LOGEMENT', 'HAS_MULTIPLE_LOGEMENTS',
#  'NO_ARROND_ILE_CUM', 'LONGITUDE', 'LATITUDE',
#  'FIRE_FREQUENCY_ZONE', 'fire', 'fire_date', 'had_fire',
#  'fire_month', 'fire_year', 'fire_season', 'year_month',
#  'FIRE_COUNT_LAST_YEAR_ZONE', 'BUILDING_COUNT',
#  'FIRE_RATE_ZONE', 'FIRE_COUNT_LAST_YEAR_ZONE_NORM', 'FIRE_RATE_ZONE_NORM',
#  'missing_coords'
# ]
# 
# 
# 
# 


data.info()



data.columns.tolist()


columns_to_drop = [
    # 🔁 Redundant address information
    "CIVIQUE_DEBUT", "CIVIQUE_FIN", "NOM_RUE", "NOM_RUE_CLEAN", "ADDR_DE",
    
    # 🆔 Internal or irrelevant identifiers
    "MATRICULE83", "LETTRE_DEBUT", "LETTRE_FIN",
    
    # 🏢 Administrative or metadata
    "SUITE_DEBUT", "CASERNE",
    
    # 🗺️ Coordinates that are missing (if not used)
    # Keep LATITUDE/LONGITUDE unless you plan to use NO_ARROND_ILE_CUM only

    # 🔄 Features replaced by engineered versions
    "ANNEE_CONSTRUCTION",  # Keep AGE_BATIMENT instead
]




# Drop columns
cleaned_data = data.drop(columns=columns_to_drop, errors='ignore')




from pathlib import Path

# 🔧 Reuse ROOT definition from earlier
ROOT = Path(__file__).resolve().parents[1]


# 📁 Define the output path relative to project structure
OUTPUT_PATH = ROOT / "datasets" / "cleaned" / "evaluation_fire_coordinates_date_feat_eng_2.csv"

# ✅ Create the output directory if it doesn't exist
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 💾 Save the cleaned DataFrame
cleaned_data.to_csv(OUTPUT_PATH, index=False)

print(f"✅ File saved to: {OUTPUT_PATH}")






