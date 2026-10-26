"""Train a basline model and a gradient boosting model using time-based split and compare prediction accuracy"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import load_data

FEATURES = [
    "season", "month", "hour", "holiday", "weekday", "workingday",
    "weather", "temp", "atemp", "humidity", "windspeed",
]
TARGET = "count"
CUTOFF = pd.Timestamp("2012-07-01")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")


def time_split(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF):
    """Train on rows before the cutoff, test on rows from the cutoff onwards."""
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]
    return train, test


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict the training mean for each (hour, workingday) combination."""
    means = train.groupby(["hour", "workingday"])[TARGET].mean()
    keys = pd.MultiIndex.from_frame(test[["hour", "workingday"]])
    return means.reindex(keys).to_numpy()


def main() -> None:
    df = load_data()
    train, test = time_split(df)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    results = {
        "baseline": evaluate(test[TARGET], baseline_predict(train, test))
    }
    # builds many small decision trees - correcting errors of previous ones - no feature scaling required
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42
    )
    model.fit(train[FEATURES], train[TARGET])
    results["gradient_boosting"] = evaluate(
        test[TARGET], model.predict(test[FEATURES])
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
