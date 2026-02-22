#To run this code Run python .\dataprep\interventions_HAS_FIRE_binary_analysis.py

import pandas as pd
import numpy as np
from datetime import datetime

DESTINATION_FILE_NAME = "./datasets/cleaned/interventions_cleaned_with_has_fire.csv"
ORIGINAL_FILE_NAME_2023_2025 = "./datasets/raw/donneesouvertes-interventions-sim.csv"
ORIGINAL_FILE_NAME_2022_BEFORE = "./datasets/raw/donneesouvertes-interventions-sim2020.csv"


def is_date_format(string_input: str, date_format: str) -> bool:
    try:
        datetime.strptime(string_input, date_format)
        return True
    except ValueError:
        return False


def convert_date_format(date_string: str) -> str:
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%d")
        return date_obj.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        try:
            datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
            return date_string
        except ValueError:
            return date_string


print("Loading data...")
df = pd.read_csv(ORIGINAL_FILE_NAME_2023_2025)
df_old = pd.read_csv(ORIGINAL_FILE_NAME_2022_BEFORE)
df = pd.concat([df, df_old])

print("Fixing date formats...")
df["CREATION_DATE_TIME"] = df["CREATION_DATE_TIME"].apply(convert_date_format)
df["CREATION_DATE_TIME"] = df["CREATION_DATE_TIME"].apply(datetime.fromisoformat)

print("Dropping unnecessary columns...")
df = df.drop(["MTM8_X", "MTM8_Y"], axis=1)

fire_categories = ["AUTREFEU", "INCENDIE"]

print("Filtering fire incidents...")
df = df[df["DESCRIPTION_GROUPE"].isin(fire_categories)]

fire_incident_count = len(df)
print(f"Total fire incidents: {fire_incident_count:,}")

category_counts = df["DESCRIPTION_GROUPE"].value_counts()
print("Fire incident breakdown by type:")
print(category_counts)

missing_summary = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame(
    {
        "Missing Count": missing_summary,
        "Missing %": missing_percentage,
    }
)

missing_report = missing_report[missing_report["Missing Count"] > 0]

print("Missing values summary:")
print(missing_report)

missing_units = df["NOMBRE_UNITES"].isnull().sum()
total_rows = len(df)
missing_pct = (missing_units / total_rows) * 100

print(f"Missing NOMBRE_UNITES values: {missing_units:,} out of {total_rows:,}")
print(f"Missing percentage: {missing_pct:.2f}%")

print("Saving cleaned dataset...")
df.to_csv(DESTINATION_FILE_NAME, index=False)

print(f"Dataset saved to {DESTINATION_FILE_NAME} with {len(df):,} fire incident records.")