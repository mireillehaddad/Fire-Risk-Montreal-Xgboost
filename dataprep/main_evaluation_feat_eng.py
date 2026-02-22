# To run: python ./dataprep/main_evaluation_feat_eng.py

import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# -------------------------------
# Load Data
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = ROOT / "datasets" / "cleaned" / "eval_cleaned_feat_eng.csv"
ADDR_PATH = ROOT / "datasets" / "cleaned" / "adresses.csv"
INC_PATH = ROOT / "datasets" / "cleaned" / "interventions_cleaned_with_has_fire.csv"

print("[eval exists?]", EVAL_PATH.exists(), "\t->", EVAL_PATH)
print("[addr exists?]", ADDR_PATH.exists(), "\t->", ADDR_PATH)
print("[inc exists?] ", INC_PATH.exists(), "\t->", INC_PATH)

if not EVAL_PATH.exists():
    raise FileNotFoundError(f"EVAL file not found at {EVAL_PATH}")
if not ADDR_PATH.exists():
    raise FileNotFoundError(f"Address file not found at {ADDR_PATH}")
if not INC_PATH.exists():
    raise FileNotFoundError(f"Interventions file not found at {INC_PATH}")


# -------------------------------
# Preprocessing
# -------------------------------
eval_df = pd.read_csv(EVAL_PATH, dtype=str)
addr_df = pd.read_csv(ADDR_PATH, dtype=str)
inc_df = pd.read_csv(INC_PATH)

eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = (
    eval_df["NOM_RUE"]
    .str.extract(r"^(.*?)(?:\s+\(.*)?$")[0]
    .str.lower()
    .str.strip()
)
original_eval_df = eval_df.copy()

addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = (
    addr_df["GENERIQUE"].str.lower().str.strip()
    + " "
    + addr_df["SPECIFIQUE"].str.lower().str.strip()
)

eval_with_coords = pd.merge(
    eval_df,
    addr_df,
    left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
    right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
    how="left",
)

eval_with_coords = eval_with_coords.dropna(subset=["LONGITUDE", "LATITUDE"])

eval_gdf = gpd.GeoDataFrame(
    eval_with_coords,
    geometry=gpd.points_from_xy(
        eval_with_coords["LONGITUDE"].astype(float),
        eval_with_coords["LATITUDE"].astype(float),
    ),
    crs="EPSG:4326",
)


# -------------------------------
# Clean and Filter Fire Incidents
# -------------------------------
inc_df = inc_df[inc_df["DESCRIPTION_GROUPE"].str.contains("INCENDIE", case=False, na=False)]
inc_df["CREATION_DATE_TIME"] = pd.to_datetime(inc_df["CREATION_DATE_TIME"], errors="coerce")

incident_gdf = gpd.GeoDataFrame(
    inc_df,
    geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
    crs="EPSG:4326",
)


# -------------------------------
# Spatial Join (100m)
# -------------------------------
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = incident_gdf.to_crs(epsg=32188)

incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")

joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate="within", how="inner")
joined = joined.rename(columns={"CREATION_DATE_TIME": "fire_date"})
joined["fire"] = True

drop_cols = [
    "CIVIQUE_DEBUT",
    "CIVIQUE_FIN",
    "NOM_RUE",
    "LETTRE_DEBUT",
    "LETTRE_FIN",
    "MATRICULE83",
    "NOM_RUE_CLEAN",
    "ADDR_DE",
    "X",
    "Y",
    "geometry",
    "geometry_right",
    "index_right",
    "DESCRIPTION_GROUPE",
    "INCIDENT_TYPE_DESC",
    "DIVISION",
    "NOM_VILLE",
    "NOM_ARROND",
]
joined.drop(columns=drop_cols, inplace=True, errors="ignore")

fire_records = joined[["ID_UEV", "fire_date", "NOMBRE_UNITES", "CASERNE"]].copy()
fire_records["fire"] = True


# -------------------------------
# Merge Fire Info Back to Evaluation
# -------------------------------
data = pd.merge(original_eval_df, fire_records, on="ID_UEV", how="left")
data["fire"] = data["fire"].fillna(False)
data["fire_date"] = pd.to_datetime(data["fire_date"], errors="coerce")

addr_coords = addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]]
data = pd.merge(
    data,
    addr_coords,
    left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
    right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
    how="left",
)


# -------------------------------
# Time Features
# -------------------------------
data["fire_month"] = data["fire_date"].dt.month
data["fire_year"] = data["fire_date"].dt.year


def get_season(month):
    if pd.isnull(month):
        return None
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"


data["fire_season"] = data["fire_month"].apply(get_season)
data["year_month"] = data["fire_date"].dt.to_period("M").astype(str)


# -------------------------------
# Zone-Level Aggregates
# -------------------------------
data["NO_ARROND_ILE_CUM"] = data["NO_ARROND_ILE_CUM"].astype(str)

fires_2024 = data[(data["fire"] == True) & (data["fire_date"].dt.year == 2024)].copy()
fire_count = fires_2024.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="FIRE_COUNT_LAST_YEAR_ZONE")
building_count = data.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="BUILDING_COUNT")

data = data.merge(fire_count, on="NO_ARROND_ILE_CUM", how="left")
data = data.merge(building_count, on="NO_ARROND_ILE_CUM", how="left")

data["FIRE_COUNT_LAST_YEAR_ZONE"] = data["FIRE_COUNT_LAST_YEAR_ZONE"].fillna(0)
data["FIRE_RATE_ZONE"] = (data["FIRE_COUNT_LAST_YEAR_ZONE"] / data["BUILDING_COUNT"]).fillna(0)

scaler = MinMaxScaler()
data[["FIRE_COUNT_LAST_YEAR_ZONE_NORM", "FIRE_RATE_ZONE_NORM"]] = scaler.fit_transform(
    data[["FIRE_COUNT_LAST_YEAR_ZONE", "FIRE_RATE_ZONE"]]
)


# -------------------------------
# Optional Validation Flags
# -------------------------------
num_missing_coords = data[["LATITUDE", "LONGITUDE"]].isna().any(axis=1).sum()
print(f"Rows with missing coordinates: {num_missing_coords}")

data["had_fire"] = data["fire_date"].notna().astype(int)

print("Final summary:")
print(f"Total rows: {len(data)}")
#print(f"Rows with fire: {int(data['fire'].sum())}")
#print(f"Rows without fire: {int((~data['fire']).sum())}")

data["fire"] = data["fire"].astype(bool)

print(f"Rows with fire: {data['fire'].sum()}")
print(f"Rows without fire: {(data['fire'] == False).sum()}")

print(f"Rows with fire date: {int(data['fire_date'].notna().sum())}")
print(f"Rows with had_fire: {int(data['had_fire'].sum())}")
print(f"Columns available: {len(data.columns)}")

data["missing_coords"] = data[["LATITUDE", "LONGITUDE"]].isna().any(axis=1)

fire_by_coords = data.groupby("missing_coords")["fire"].value_counts(normalize=True).unstack().fillna(0)
print("Fire distribution by coordinate presence:")
print(fire_by_coords)


# -------------------------------
# Feature Selection and Save
# -------------------------------
columns_to_drop = [
    "CIVIQUE_DEBUT",
    "CIVIQUE_FIN",
    "NOM_RUE",
    "NOM_RUE_CLEAN",
    "ADDR_DE",
    "MATRICULE83",
    "LETTRE_DEBUT",
    "LETTRE_FIN",
    "SUITE_DEBUT",
    "CASERNE",
    "ANNEE_CONSTRUCTION",
]

cleaned_data = data.drop(columns=columns_to_drop, errors="ignore")

OUTPUT_PATH = ROOT / "datasets" / "cleaned" / "evaluation_fire_coordinates_date_feat_eng_2.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cleaned_data.to_csv(OUTPUT_PATH, index=False)

print(f"File saved to: {OUTPUT_PATH}")