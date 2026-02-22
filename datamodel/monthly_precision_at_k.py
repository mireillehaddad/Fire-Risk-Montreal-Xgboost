from pathlib import Path
import pandas as pd


def monthly_precision_at_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    For each month:
      - rank rows by predicted_proba desc
      - take top-k
      - compute precision@k = (# true fires in top-k) / k
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    results = []
    for month, g in df.groupby("month", dropna=True):
        g = g.sort_values("predicted_proba", ascending=False)
        topk = g.head(k)

        if len(topk) == 0:
            continue

        fires_in_topk = int(topk["HAS_FIRE_THIS_MONTH"].sum())
        precision = fires_in_topk / len(topk)

        results.append({
            "month": month.strftime("%Y-%m"),
            "k": k,
            "precision_at_k": precision,
            "fires_in_top_k": fires_in_topk,
            "rows_in_month": len(g),
        })

    return pd.DataFrame(results)


def main():
    ROOT = Path(__file__).resolve().parents[1]

    INPUT_PRED_CSV = ROOT / "datamodel" / "xgb_time_model_predictions.csv"
    OUTPUT_MONTHLY = ROOT / "datamodel" / "monthly_precision_at_k.csv"
    OUTPUT_AVG = ROOT / "datamodel" / "avg_precision_at_k.csv"

    print("Input:", INPUT_PRED_CSV)
    if not INPUT_PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions file: {INPUT_PRED_CSV}")

    df = pd.read_csv(INPUT_PRED_CSV)

    required = {"month", "HAS_FIRE_THIS_MONTH", "predicted_proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in prediction CSV: {missing}")

    k_values = [100, 500, 1000, 5000, 10000]

    all_monthly = []
    for k in k_values:
        res_k = monthly_precision_at_k(df, k)
        all_monthly.append(res_k)

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    monthly_df.to_csv(OUTPUT_MONTHLY, index=False)

    avg_df = (
        monthly_df.groupby("k", as_index=False)["precision_at_k"]
        .mean()
        .rename(columns={"precision_at_k": "avg_precision_at_k"})
    )
    avg_df.to_csv(OUTPUT_AVG, index=False)

    print("Saved monthly results:", OUTPUT_MONTHLY)
    print("Saved averages:", OUTPUT_AVG)
    print("\nAverages:")
    print(avg_df)


if __name__ == "__main__":
    main()