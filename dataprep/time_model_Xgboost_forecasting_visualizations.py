#Run the script (example name)
#python  .\dataprep\time_model_Xgboost_forecasting_visualizations.py

#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    fbeta_score,
)

import folium
from folium.plugins import HeatMap


def get_project_root() -> Path:
    """
    Returns the project root folder.
    Assumes this script is executed from inside the repository (any subfolder is fine).
    """
    return Path.cwd()


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def evaluate_thresholds(y_true, y_probs, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("f2", ascending=False).reset_index(drop=True)


def build_precision_at_k(result_df: pd.DataFrame, k_values, proba_col="predicted_proba", target_col="target"):
    """
    Precision@k on a ranked list.
    Precision@k = (# true positives in top k) / k
    """
    ranked = result_df.sort_values(proba_col, ascending=False).reset_index(drop=True)
    out = []
    for k in k_values:
        topk = ranked.head(k)
        precision_k = topk[target_col].sum() / float(k)
        out.append({"k": int(k), "precision_at_k": float(precision_k), "positives_in_top_k": int(topk[target_col].sum())})
    return pd.DataFrame(out)


def main():
    ROOT = get_project_root()
    print(f"Root folder: {ROOT}")

    # Paths
    INPUT_PANEL = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"
    OUTPUT_PRED = ROOT / "datamodel" / "xgb_time_model_predictions.csv"

    print("[input exists?]", INPUT_PANEL.exists(), "->", INPUT_PANEL)

    # Load panel
    df = safe_read_csv(INPUT_PANEL)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    else:
        raise ValueError("Expected column 'month' not found in panel dataset.")

    df = df.sort_values(["ID_UEV", "month"]).copy()
    df["year"] = df["month"].dt.year

    # Add lag features if not already present
    for lag in range(1, 4):
        col = f"fire_last_{lag}m"
        if col not in df.columns:
            df[col] = (
                df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
                .shift(lag)
                .fillna(0)
                .astype(int)
            )

    # Features and target
    features = [
        "MUNICIPALITE", "ETAGE_HORS_SOL", "NOMBRE_LOGEMENT", "AGE_BATIMENT",
        "CODE_UTILISATION", "CATEGORIE_UEF", "SUPERFICIE_TERRAIN", "SUPERFICIE_BATIMENT",
        "NO_ARROND_ILE_CUM", "RATIO_SURFACE", "DENSITE_LOGEMENT", "HAS_MULTIPLE_LOGEMENTS",
        "FIRE_FREQUENCY_ZONE", "FIRE_RATE_ZONE", "FIRE_COUNT_LAST_YEAR_ZONE", "BUILDING_COUNT",
        "FIRE_RATE_ZONE_NORM", "FIRE_COUNT_LAST_YEAR_ZONE_NORM",
        "fire_last_1m", "fire_last_2m", "fire_last_3m",
        "fire_cumcount", "fire_rolling_3m", "fire_rolling_6m", "fire_rolling_12m",
        "month_num", "year",
    ]
    target = "HAS_FIRE_THIS_MONTH"

    # Ensure required columns exist
    missing_cols = [c for c in features + [target] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Encode categorical variables
    encoders = {}
    for col in ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df["CATEGORIE_UEF"] = df["CATEGORIE_UEF"].astype("category")
    df["NO_ARROND_ILE_CUM"] = df["NO_ARROND_ILE_CUM"].astype("category")

    # Train/Test split
    train_df = df[df["year"] <= 2023].copy()
    test_df = df[df["year"] == 2024].copy()

    X_train = train_df[features]
    y_train = train_df[target].astype(int)

    X_test = test_df[features]
    y_test = test_df[target].astype(int)

    print("Training features:", X_train.columns.tolist())

    # Class imbalance weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0:
        raise ValueError("No positive samples in training set.")
    scale_pos_weight = neg / pos

    # Train model
    print("Training model ...")
    model = XGBClassifier(
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate default threshold 0.5
    print("Evaluating model at default threshold 0.5 ...")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, digits=3))

    # Precision-Recall vs threshold plot
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate a few thresholds + F2
    threshold_list = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    threshold_df = evaluate_thresholds(y_test, y_probs, threshold_list)
    print("\nThreshold metrics (sorted by F2):")
    print(threshold_df)

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f2"], marker="o", label="F2")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F2 vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion matrix at best F2 threshold
    best_t = float(threshold_df.iloc[0]["threshold"])
    y_pred_best = (y_probs >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fire", "Fire"],
        yticklabels=["No Fire", "Fire"],
    )
    plt.title(f"Confusion Matrix (threshold = {best_t})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Save predictions to CSV
    result_test = test_df.copy(deep=True)
    result_test["predicted_proba"] = y_probs
    result_test["predicted_result"] = y_pred_best
    result_test["target"] = y_test.values
    result_test.to_csv(OUTPUT_PRED, index=False)
    print(f"Saved test set predictions to {OUTPUT_PRED}")

    # Precision@k on ranked predictions
    k_values = [100, 500, 1000, 5000, 10000]
    p_at_k = build_precision_at_k(result_test, k_values, proba_col="predicted_proba", target_col="target")
    print("\nPrecision@k:")
    print(p_at_k)

    # Optional: simple folium heatmap using coordinates (if present)
    if "LATITUDE" in result_test.columns and "LONGITUDE" in result_test.columns:
        map_df = result_test.dropna(subset=["LATITUDE", "LONGITUDE", "predicted_proba"]).copy()
        if not map_df.empty:
            center = [map_df["LATITUDE"].astype(float).mean(), map_df["LONGITUDE"].astype(float).mean()]
            m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

            # Use top risk points for visualization
            top_n = map_df.sort_values("predicted_proba", ascending=False).head(50000)
            heat_data = top_n[["LATITUDE", "LONGITUDE", "predicted_proba"]].astype(float).values.tolist()

            HeatMap(
                heat_data,
                radius=8,
                blur=12,
                max_zoom=12,
                gradient={0.3: "yellow", 0.6: "orange", 1.0: "red"},
            ).add_to(m)

            out_html = ROOT / "datamodel" / "fire_risk_heatmap.html"
            m.save(str(out_html))
            print(f"Saved heatmap to {out_html}")


if __name__ == "__main__":
    main()