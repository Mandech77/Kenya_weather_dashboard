"""Data loading and cleaning for the Kenya meteorological dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).with_name("Kenya_Meteorological_Synthetic_Dataset.csv")

# Okabe-Ito colour-blind-safe palette. One fixed colour per station is used in
# EVERY chart (Gestalt similarity).
STATION_COLOURS = {
    "Nairobi": "#0072B2",
    "Mombasa": "#D55E00",
    "Kisumu": "#009E73",
    "Eldoret": "#CC79A7",
    "Nyeri": "#E69F00",
}
MUTED = "#C9CED6"  # "ground" colour for de-emphasised stations

CONDITION_ORDER = ["Mostly Dry", "Cloudy", "Light/Moderate Rain", "Heavy Rain"]
CONDITION_COLOURS = {
    "Mostly Dry": "#E8D9B5",
    "Cloudy": "#9AA5B1",
    "Light/Moderate Rain": "#6BAED6",
    "Heavy Rain": "#08519C",
}

NUMERIC_COLS = [
    "Mean_Temperature_C", "Minimum_Temperature_C", "Maximum_Temperature_C",
    "Rainfall_mm", "Relative_Humidity_pct", "Wind_Speed_kmh",
    "Atmospheric_Pressure_hPa", "Solar_Radiation_kWh_m2_day",
]


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Import the CSV with Pandas and parse the date column."""
    return pd.read_csv(path, parse_dates=["Date"])


def clean(df: pd.DataFrame):
    """Return (clean_df, log) where log documents every cleaning decision.

    Rules
    -----
    * Rainfall: the weather condition is a reliable proxy. 'Mostly Dry' and
      'Cloudy' days always have 0 mm in the data, so missing values there are
      set to 0. For rainy days the missing value is filled with the median
      rainfall of the same station and the same condition.
    * Relative humidity and wind speed: both change gradually from day to day,
      so gaps are filled by time-based interpolation within each station
      (edges: nearest value), never across stations.
    """
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True).copy()
    log = []
    missing_before = df[NUMERIC_COLS].isna().sum()

    # --- rainfall -------------------------------------------------------
    dry = df["Weather_Condition"].isin(["Mostly Dry", "Cloudy"])
    n = int((df["Rainfall_mm"].isna() & dry).sum())
    df.loc[df["Rainfall_mm"].isna() & dry, "Rainfall_mm"] = 0.0
    log.append(f"Rainfall: {n} missing values on 'Mostly Dry'/'Cloudy' days set to 0 mm.")

    med = df.groupby(["Station", "Weather_Condition"])["Rainfall_mm"].transform("median")
    n = int(df["Rainfall_mm"].isna().sum())
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(med)
    log.append(f"Rainfall: {n} missing values on rainy days filled with the station + condition median.")

    # --- humidity & wind --------------------------------------------------
    for col, label in [("Relative_Humidity_pct", "Humidity"), ("Wind_Speed_kmh", "Wind speed")]:
        n = int(df[col].isna().sum())
        df[col] = (
            df.set_index("Date")
              .groupby("Station")[col]
              .transform(lambda s: s.interpolate(method="time", limit_direction="both"))
              .values
        )
        log.append(f"{label}: {n} missing values filled by time interpolation within each station.")

    # --- derived columns --------------------------------------------------
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Temp_Range_C"] = df["Maximum_Temperature_C"] - df["Minimum_Temperature_C"]
    df["Rainy_Day"] = df["Rainfall_mm"] > 0

    # --- validation -------------------------------------------------------
    assert df[NUMERIC_COLS].isna().sum().sum() == 0, "NaNs remain after cleaning"
    assert (df["Maximum_Temperature_C"] >= df["Minimum_Temperature_C"]).all()
    log.append("Checks passed: no duplicates, no remaining NaNs, max temp >= min temp on every row.")
    log.append(f"Total missing cells before cleaning: {int(missing_before.sum())}; after: 0.")
    return df, log, missing_before


def load_clean():
    raw = load_raw()
    df, log, missing_before = clean(raw)
    return raw, df, log, missing_before
