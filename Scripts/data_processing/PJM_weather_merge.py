#!/usr/bin/env python3
import warnings
import pandas as pd
import numpy as np
import holidays
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# PJM = pd.read_csv("Data/intermediate/PJM_intermediate.csv")
PJM = pd.read_parquet("Data/intermediate/PJM_intermediate.parquet")
PJM["datetime_beginning_utc"] = pd.to_datetime(PJM["datetime_beginning_utc"])
PJM["date"] = PJM["datetime_beginning_utc"].dt.floor("D")

# weather = pd.read_csv("Data/raw/weather_hist.csv")
weather = pd.read_parquet("Data/raw/weather_hist.parquet")
weather["date"] = pd.to_datetime(weather["date"])

merged = PJM.merge(weather, on="date", how="left")

#further feature engineering
df = merged 

df["season_year"] = np.where(df["Month"] >= 8,
                                df["Year"],
                                df["Year"] - 1)

df["is_weekend"] = df["Day_of_Week"].isin([5,6]).astype(int)

us_holidays = holidays.US()
df["is_holiday"] = df["date"].isin(us_holidays).astype(int)

#sqrt and cubic of basic numericals
df["mw_lag_1y_sq"] = df["mw_lag_1y"] ** 2
df["mw_lag_1y_cu"] = df["mw_lag_1y"] ** 3

df["prev_month_mean_mw_sq"] = df["prev_month_mean_mw"] ** 2
df["prev_month_mean_mw_cu"] = df["prev_month_mean_mw"] ** 3


# find all min/max/mean weather columns automatically
weather_cols = [col for col in df.columns 
                if col.startswith(("min_", "max_", "mean_"))]

#hinge temp cooling / heating 
c = 15.56  

mean_cols = [col for col in df.columns if col.startswith("mean_")]

for col in mean_cols:
    df[f"{col}_below_60"] = np.maximum(0, c - df[col])
    df[f"{col}_below_60_sq"] = np.maximum(0, c - df[col]) ** 2
    df[f"{col}_above_60"] = np.maximum(0, df[col] - c)
    df[f"{col}_above_60_sq"] = np.maximum(0, df[col] - c) ** 2

#sqrt weather
for col in weather_cols:
    df[f"{col}_sq"] = df[col] ** 2
    df[f"{col}_cu"] = df[col] ** 3

# df.to_csv("Data/processed/full_data.csv", index=False)
df.to_parquet("Data/processed/full_data.parquet", index=False)
