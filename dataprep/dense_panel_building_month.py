# To run this code : python ./dataprep/dense_panel_building_month.py
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from xgboost import XGBClassifier
from sklearn.metrics import classification_report




from pathlib import Path
import os

# 🔧 Project root directory (2 levels up from current script)
ROOT = Path(__file__).parents[1]

# 🔹 Define input/output paths using ROOT
INPUT_CSV = ROOT / "datasets" / "cleaned" / "evaluation_fire_coordinates_date_feat_eng_2.csv"
OUTPUT_PANEL = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"

# 🔍 Optional: check existence
print("[input exists?]", INPUT_CSV.exists(), "➜", INPUT_CSV)
print("[output dir exists?]", OUTPUT_PANEL.parent.exists(), "➜", OUTPUT_PANEL.parent)





# 🔸 Load and clean fire dataset
df = pd.read_csv(INPUT_CSV)
df.head()





print(f"🧮 Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")









#clean data
df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df["month"] = df["fire_date"].dt.to_period("M")
df = df.dropna(subset=["LONGITUDE", "LATITUDE", "ID_UEV"])
df["geometry"] = df.apply(lambda row: Point(row["LONGITUDE"], row["LATITUDE"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:32188")






# # 🔹 Create dense panel: all buildings × all months



unique_buildings = gdf[["ID_UEV", "LATITUDE", "LONGITUDE"]].drop_duplicates()
all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
print(f"Expanding dataset into a panel of building x month ...")
panel = pd.MultiIndex.from_product(
    [unique_buildings["ID_UEV"].unique(), all_months],
    names=["ID_UEV", "month"]
).to_frame(index=False)
panel = panel.merge(unique_buildings, on="ID_UEV", how="left")





# # 🔹 Label fire presence
# fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
# fires["HAS_FIRE_THIS_MONTH"] = 1
# panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
# panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)



# print("Adding time features ...")
# # 🔹 Add time features
# panel["month_num"] = panel["month"].dt.month
# panel["year"] = panel["month"].dt.year





print(f"Base Panel shape: {panel.shape}")








#Extract full static features from gdf
# Define static columns to retain in the panel (no time-dependency)
static_cols = [
    "ID_UEV", "LATITUDE", "LONGITUDE", "MUNICIPALITE", "ETAGE_HORS_SOL",
    "NOMBRE_LOGEMENT", "AGE_BATIMENT", "CODE_UTILISATION", "CATEGORIE_UEF",
    "SUPERFICIE_TERRAIN", "SUPERFICIE_BATIMENT", "NO_ARROND_ILE_CUM", "RATIO_SURFACE",
    "DENSITE_LOGEMENT", "HAS_MULTIPLE_LOGEMENTS", "FIRE_FREQUENCY_ZONE",
    "FIRE_RATE_ZONE", "FIRE_COUNT_LAST_YEAR_ZONE", "BUILDING_COUNT",
    "FIRE_RATE_ZONE_NORM", "FIRE_COUNT_LAST_YEAR_ZONE_NORM"
]

print("Merging static building features ...")
# Drop duplicates so each building has one row of static info
static_features = gdf[static_cols].drop_duplicates(subset=["ID_UEV"])

# panel = panel.merge(static_features, on="ID_UEV", how="left")

# print(f"Static features merged - panel shape:  {panel.shape}")


# print(panel.columns.tolist())










# ✅ Recommended Adjustments
# 1. Filter static_features["ID_UEV"] to only valid buildings
# If static_features was created from a larger dataset, it may contain IDs not in your final cleaned gdf. Filter it first:



print(f"Filtering static features with valid ID")
valid_ids = gdf["ID_UEV"].unique()
static_features = static_features[static_features["ID_UEV"].isin(valid_ids)]





#Create panel and merge with static_features
# Build the panel with only valid building IDs
all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
panel = pd.MultiIndex.from_product(
    [static_features["ID_UEV"].unique(), all_months],
    names=["ID_UEV", "month"]
).to_frame(index=False)





# Merge cleanly with static features
panel = panel.merge(static_features, on="ID_UEV", how="left")

print(f"static features merged  - shape= {panel.shape}")












# 🔸 Construct panel: building × month
#unique_buildings = gdf[["ID_UEV", "LATITUDE", "LONGITUDE"]].drop_duplicates()
#all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
#panel = pd.MultiIndex.from_product([unique_buildings["ID_UEV"], all_months],
#                                   names=["ID_UEV", "month"]).to_frame(index=False)
#panel = panel.merge(unique_buildings, on="ID_UEV", how="left")




# # 🔸 1. Label fire presence
print("Labelling HAS_FIRE_THIS_MONTH")
fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
fires["HAS_FIRE_THIS_MONTH"] = 1
panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)

# 🔸 2. Create lag features
print("Adding last month(s) fire presence ...")
panel = panel.sort_values(by=["ID_UEV", "month"])
panel["fire_last_1m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(1).fillna(0)
panel["fire_last_2m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(2).fillna(0)
panel["fire_last_3m"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].shift(3).fillna(0)



# ✅ What We’ll Add
# fire_cumcount: Total number of past fires up to (but not including) this month.
# 
# fire_rolling_3m: Number of fires in the last 3 full months.
# 
# fire_rolling_6m: Number of fires in the last 6 full months.
# 
# (Optional) fire_rolling_12m: Fires in the last year.



# ✅ Sort panel by ID_UEV and month (ascending)
panel = panel.sort_values(by=["ID_UEV", "month"]).reset_index(drop=True)



print("adding cumulative fire columns ...")
# 🔄 Sort before group-based ops
panel = panel.sort_values(by=["ID_UEV", "month"])

panel["fire_cumcount"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().cumsum())
    .fillna(0)
)


# 🔄 Rolling fire count: 3 months
panel["fire_rolling_3m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=3, min_periods=1).sum())
    .fillna(0)
)

# 🔄 Rolling fire count: 6 months
panel["fire_rolling_6m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=6, min_periods=1).sum())
    .fillna(0)
)

# Optional: Rolling 12 months
panel["fire_rolling_12m"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().rolling(window=12, min_periods=1).sum())
    .fillna(0)
)


# 🚀 Recommended Next Steps
# Add more temporal signals:
# 
# has_fire_last_month (binary from shift(1))
# 
# has_fire_2_months_ago, etc.
# 
# months_since_last_fire

# In[17]:


panel["has_fire_last_month"] = panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"].transform(lambda x: x.shift(1).fillna(0))
panel["months_since_last_fire"] = (
    panel.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
    .transform(lambda x: x.shift().apply(lambda val: 0 if val == 1 else None).ffill(limit=None).cumsum())
    .fillna(999)  # or a max cap for "never had fire"
)


# In[18]:


# 🔸 Add time-based features
print("Adding time features month and year")
panel["month_num"] = panel["month"].dt.month
panel["year"] = panel["month"].dt.year


# In[19]:


print(f"Final panel shape : {panel.shape}")
print(panel.columns)





# 💾 Save panel
panel.to_csv(OUTPUT_PANEL, index=False)
print(f"✅ Panel saved to {OUTPUT_PANEL}")

# print("Saving to:", OUTPUT_PANEL.resolve())



