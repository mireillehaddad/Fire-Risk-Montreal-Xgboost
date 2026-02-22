# To run: python ./dataprep/dense_panel_building_month.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

ROOT = Path(__file__).parents[1]

INPUT_CSV = ROOT / "datasets" / "cleaned" / "evaluation_fire_coordinates_date_feat_eng_2.csv"
OUTPUT_PANEL = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"

print("[input exists?]", INPUT_CSV.exists(), "->", INPUT_CSV)
print("[output dir exists?]", OUTPUT_PANEL.parent.exists(), "->", OUTPUT_PANEL.parent)

df = pd.read_csv(INPUT_CSV)

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df["month"] = df["fire_date"].dt.to_period("M")

df = df.dropna(subset=["LONGITUDE", "LATITUDE", "ID_UEV"])

df["geometry"] = df.apply(lambda row: Point(row["LONGITUDE"], row["LATITUDE"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:32188")

static_cols = [
    "ID_UEV",
    "LATITUDE",
    "LONGITUDE",
    "MUNICIPALITE",
    "ETAGE_HORS_SOL",
    "NOMBRE_LOGEMENT",
    "AGE_BATIMENT",
    "CODE_UTILISATION",
    "CATEGORIE_UEF",
    "SUPERFICIE_TERRAIN",
    "SUPERFICIE_BATIMENT",
    "NO_ARROND_ILE_CUM",
    "RATIO_SURFACE",
    "DENSITE_LOGEMENT",
    "HAS_MULTIPLE_LOGEMENTS",
    "FIRE_FREQUENCY_ZONE",
    "FIRE_RATE_ZONE",
    "FIRE_COUNT_LAST_YEAR_ZONE",
    "BUILDING_COUNT",
    "FIRE_RATE_ZONE_NORM",
    "FIRE_COUNT_LAST_YEAR_ZONE_NORM",
]

print("Building static features table...")
static_features = gdf[static_cols].drop_duplicates(subset=["ID_UEV"])

valid_ids = gdf["ID_UEV"].unique()
static_features = static_features[static_features["ID_UEV"].isin(valid_ids)]

all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")

print("Expanding dataset into a building x month panel...")
panel = pd.MultiIndex.from_product(
    [static_features["ID_UEV"].unique(), all_months],
    names=["ID_UEV", "month"],
).to_frame(index=False)

print("Merging static building features...")
panel = panel.merge(static_features, on="ID_UEV", how="left")
print(f"Static features merged - shape: {panel.shape}")

print("Labelling HAS_FIRE_THIS_MONTH...")
fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
fires["HAS_FIRE_THIS_MONTH"] = 1
panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)

print("Adding lag features...")
panel = panel.sort_values(by=["ID_UEV", "month"])
panel["fire_last_1m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(1).fillna(0)
panel["fire_last_2m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(2).fillna(0)
panel["fire_last_3m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(3).fillna(0)

print("Adding cumulative and rolling fire features...")
panel = panel.sort_values(by=["ID_UEV", "month"]).reset_index(drop=True)

panel["fire_cumcount"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().cumsum())
    .fillna(0)
)

panel["fire_rolling_3m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=3, min_periods=1).sum())
    .fillna(0)
)

panel["fire_rolling_6m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=6, min_periods=1).sum())
    .fillna(0)
)

panel["fire_rolling_12m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=12, min_periods=1).sum())
    .fillna(0)
)

panel["has_fire_last_month"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].transform(lambda x: x.shift(1).fillna(0))
)

print("Adding time features...")
panel["month_num"] = panel["month"].dt.month
panel["year"] = panel["month"].dt.year

print(f"Final panel shape: {panel.shape}")
print("Saving panel...")
panel.to_csv(OUTPUT_PANEL, index=False)

print(f"Panel saved to {OUTPUT_PANEL}")