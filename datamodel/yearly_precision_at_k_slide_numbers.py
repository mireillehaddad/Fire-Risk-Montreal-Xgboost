#Run python .\datamodel\yearly_precision_at_k_slide_numbers.py
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

PRED_PATH = ROOT / "datamodel" / "xgb_time_model_predictions.csv"  # adjust if different
YEAR = 2024
K = 7500
N_SIM = 300  # random baseline simulations
RANDOM_SEED = 42

# Columns - adjust if your file uses different names
COL_ID = "ID_UEV"
COL_MONTH = "month"
COL_PROBA = "predicted_proba"
COL_TARGET = "HAS_FIRE_THIS_MONTH"  # sometimes "target"

def main():
    print("Input:", PRED_PATH)
    df = pd.read_csv(PRED_PATH)

    # Ensure month is datetime
    df[COL_MONTH] = pd.to_datetime(df[COL_MONTH], errors="coerce")
    df["year"] = df[COL_MONTH].dt.year

    # Keep only the evaluation year
    df = df[df["year"] == YEAR].copy()

    # Safety checks
    needed = {COL_ID, COL_MONTH, COL_PROBA, COL_TARGET}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {missing}")

    # Convert types
    df[COL_PROBA] = pd.to_numeric(df[COL_PROBA], errors="coerce")
    df[COL_TARGET] = pd.to_numeric(df[COL_TARGET], errors="coerce").fillna(0).astype(int)

    # Build yearly building-level table
    yearly = (
        df.groupby(COL_ID)
          .agg(
              risk_year=(COL_PROBA, "max"),          # use max proba across months
              had_fire_year=(COL_TARGET, "max")      # 1 if any month had fire
          )
          .reset_index()
          .dropna(subset=["risk_year"])
    )

    n_buildings = len(yearly)
    total_fire_buildings = int(yearly["had_fire_year"].sum())
    base_rate = total_fire_buildings / n_buildings

    print(f"Buildings in {YEAR}: {n_buildings:,}")
    print(f"Buildings with >=1 fire in {YEAR}: {total_fire_buildings:,} ({base_rate:.3%})")
    print(f"Yearly inspection capacity k = {K:,}")

    if K > n_buildings:
        raise ValueError(f"K={K} is bigger than number of buildings={n_buildings}")

    # Highest predicted risk method
    topk = yearly.sort_values("risk_year", ascending=False).head(K)
    fires_topk = int(topk["had_fire_year"].sum())
    precision_topk = fires_topk / K

    print("\nMethod Comparison")
    print(f"Highest predicted risk -> Actual fire buildings in top-k: {fires_topk:,}")
    print(f"Precision@k = {precision_topk:.4f}")

    # Random pick baseline (simulate)
    rng = np.random.default_rng(RANDOM_SEED)
    random_counts = []
    idx = np.arange(n_buildings)

    had_fire_arr = yearly["had_fire_year"].to_numpy()

    for _ in range(N_SIM):
        sample_idx = rng.choice(idx, size=K, replace=False)
        random_counts.append(int(had_fire_arr[sample_idx].sum()))

    random_avg = float(np.mean(random_counts))
    random_std = float(np.std(random_counts))

    print(f"\nRandom pick baseline (avg over {N_SIM} sims) -> {random_avg:,.1f} ± {random_std:,.1f}")

    lift = fires_topk / random_avg if random_avg > 0 else np.nan
    print(f"\nLift = highest_risk / random_pick = {lift:.2f}x")

    # Optional: save a small summary CSV for slides
    out = pd.DataFrame({
        "method": ["random_pick_avg", "highest_predicted_risk"],
        "actual_fire_buildings": [random_avg, fires_topk],
        "k": [K, K],
        "year": [YEAR, YEAR]
    })
    out_path = ROOT / "datamodel" / f"yearly_precision_at_k_summary_{YEAR}_k{K}.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()