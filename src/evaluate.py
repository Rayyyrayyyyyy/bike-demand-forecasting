"""Evaluate the saved model on the test period: errors by hour, weather and time."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_data
from src.train import FEATURES, MODEL_PATH, TARGET, baseline_predict, time_split

FIG_DIR = Path("reports/figures")
ANALYSIS_PATH = Path("reports/error_analysis.json")


def mae_by_group(df: pd.DataFrame, pred_col: str, group_col: str) -> pd.Series:
    """Mean absolute error of a prediction column, grouped by another column."""
    return (df[TARGET] - df[pred_col]).abs().groupby(df[group_col]).mean()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train, test = time_split(df)
    model = joblib.load(MODEL_PATH)

    test = test.copy()
    test["baseline"] = baseline_predict(train, test)
    test["gradient_boosting"] = model.predict(test[FEATURES])

    # 1. Error by hour of day - calculate the MAE for each hr separtely for both models to help identify times of day where predictions are the least accurate
    by_hour = pd.DataFrame(
        {
            "Baseline": mae_by_group(test, "baseline", "hour"),
            "Gradient boosting": mae_by_group(test, "gradient_boosting", "hour"),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    by_hour.plot(ax=ax)
    ax.set(title="Test MAE by hour of day", 
           xlabel="Hour", 
           ylabel="MAE (bikes/hour)")
    fig.savefig(FIG_DIR / "error_by_hour.png", dpi=150, bbox_inches="tight")

    # 2. Actual vs predicted over the test period (weekly averages)
    weekly = test.set_index("datetime")[
        [TARGET, "baseline", "gradient_boosting"]
    ].resample("W").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    weekly.plot(ax=ax)
    ax.set(title="Actual vs predicted (weekly mean)", xlabel="", ylabel="Rentals/hour")
    fig.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")

    # 3. Numbers for the README
    #calculate the average prediction bias - if postive model overpredicts on average if negative model underpredicts on average
    bias = float((test["gradient_boosting"] - test[TARGET]).mean())
    by_weather = mae_by_group(test, "gradient_boosting", "weather")
    analysis = {
        "test_mean_count": float(test[TARGET].mean()),
        "train_mean_count": float(train[TARGET].mean()),
        "mean_prediction": float(test["gradient_boosting"].mean()),
        "bias_bikes_per_hour": bias,
        "worst_hours": by_hour["Gradient boosting"].nlargest(3).round(1).to_dict(),
        "mae_by_weather": by_weather.round(1).to_dict(),
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, default=str))
    print(json.dumps(analysis, indent=2, default=str))


if __name__ == "__main__":
    main()
