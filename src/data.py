"""Load and clean the UCI Bike Sharing (hourly) dataset"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/raw/hour.csv")

#loads datasets & cleans columns & returns the records in time order 
def load_data(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"])

    #combines data and hour column into one timestamp
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    #drops leakage and redundant columns
    df = df.drop(columns=["instant", "dteday", "yr", "casual", "registered"])

    #renaming columns - easier to understand
    df = df.rename(
        columns={
            "cnt": "count",
            "hr": "hour",
            "mnth": "month",
            "weathersit": "weather",
            "hum": "humidity",

         }
    )

    #sorted by time so a timed-based train/test split works later
    return df.sort_values("datetime").reset_index(drop=True)
    #drop=True arguement - old index discared instead of added to new column 

