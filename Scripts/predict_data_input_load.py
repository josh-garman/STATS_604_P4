#!/usr/bin/env python3
import time 
import warnings
import pandas as pd
import numpy as np
import holidays
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

US_HOLIDAYS = holidays.US()

def build_prediction_frame(train_df, weather_df, target_date=None):
    """
    skeleton prediction frame 
    """
    #NOTE fit this potentially !!
    if target_date is None:
        target_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).normalize()

    load_areas = sorted(train_df["load_area"].unique())
    
    rows = []
    for area in load_areas:
        for h in range(24):
            dt = target_date + pd.Timedelta(hours=h)
            rows.append(
                {
                    "datetime_beginning_utc": dt,
                    "load_area": area,
                    "Year": dt.year,
                    "Month": dt.month,
                    "Day": dt.day,
                    "Hour": dt.hour,
                    "Day_of_Week": dt.dayofweek,  # Monday=0
                }
            )

    pred = pd.DataFrame(rows)

    # date-only column for weather merge + holidays
    pred["date"] = pred["datetime_beginning_utc"].dt.date
    
    # weekend / holiday / season_year the same way as in training
    pred["is_weekend"] = pred["Day_of_Week"].isin([5, 6]).astype(int)
    pred["is_holiday"] = pred["date"].apply(lambda d: int(d in US_HOLIDAYS))

    # the timestamp one year ago for each prediction row
    pred["dt_1y_ago"] = pred["datetime_beginning_utc"] - pd.DateOffset(years=1)
    
    # take only what we need from training
    hist = train_df[
        [
            "datetime_beginning_utc",
            "load_area",
            "Hour",
            "mw"
        ]
    ]
    hist = hist.rename(
        columns={
            "mw": "mw_lag_1y" 
        }
    )
    
    pred["dt_1y_ago"] = pd.to_datetime(pred["dt_1y_ago"])
    hist["datetime_beginning_utc"] = pd.to_datetime(hist["datetime_beginning_utc"])

    # merge: prediction row ↔ the row from one year ago in that load_area
    pred = pred.merge(
        hist,
        left_on=["load_area", "dt_1y_ago"],
        right_on=["load_area", "datetime_beginning_utc"],
        how="left",
        suffixes=("", "_hist"),
    )

    #Year month 
    pred["year_month"] = pred["datetime_beginning_utc"].dt.to_period("M") - 1
    hist["year_month"] = hist["datetime_beginning_utc"].dt.to_period("M") 

    #Monthly Avg 
    monthly = (
        hist.groupby(["load_area", "Hour", "year_month"], as_index=False)["mw_lag_1y"]
        .mean()
        .rename(columns={"mw_lag_1y": "prev_month_mean_mw"})
    )

    #merge
    pred = pred.merge(
        monthly[["load_area", "Hour", "year_month", "prev_month_mean_mw"]],
        on=["load_area", "Hour", "year_month"],
        how="left"
    )

    weather_df = weather_df
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    pred = pred.merge(weather_df, on="date", how="left")

    #sqrt and cubic of basic numericals
    pred["mw_lag_1y_sq"] = pred["mw_lag_1y"] ** 2
    pred["mw_lag_1y_cu"] = pred["mw_lag_1y"] ** 3

    pred["prev_month_mean_mw_sq"] = pred["prev_month_mean_mw"] ** 2
    pred["prev_month_mean_mw_cu"] = pred["prev_month_mean_mw"] ** 3


    # find all min/max/mean weather columns automatically
    weather_cols = [col for col in pred.columns 
                    if col.startswith(("min_", "max_", "mean_"))]

    #hinge temp cooling / heating 
    c = 15.56  

    mean_cols = [col for col in pred.columns if col.startswith("mean_")]

    for col in mean_cols:
        pred[f"{col}_below_60"] = np.maximum(0, c - pred[col])
        pred[f"{col}_below_60_sq"] = np.maximum(0, c - pred[col]) ** 2
        pred[f"{col}_above_60"] = np.maximum(0, pred[col] - c)
        pred[f"{col}_above_60_sq"] = np.maximum(0, pred[col] - c) ** 2

    #sqrt weather
    for col in weather_cols:
        pred[f"{col}_sq"] = pred[col] ** 2
        pred[f"{col}_cu"] = pred[col] ** 3

    # fix datetime columns & drop temp columns
    pred = pred.drop(columns=["datetime_beginning_utc_hist", "Hour_hist", "year_month","dt_1y_ago"])

    return pred
 


# hist_data = pd.read_csv("Data/processed/full_data.csv")
usecols = ["datetime_beginning_utc", "load_area", "Hour", "mw"]

# hist_data = pd.read_csv(
#     "Data/processed/full_data.csv",
#     usecols=usecols
# )

hist_data = pd.read_parquet(
    "Data/processed/full_data.parquet",
    columns=usecols
)


# weather_df = pd.read_csv("Data/intermediate/weather_tomorrow_forecast.csv")
weather_df = pd.read_parquet("Data/intermediate/weather_tomorrow_forecast.parquet")
pred_frame = build_prediction_frame(hist_data, weather_df=weather_df, target_date=None)
pred_frame.to_csv("Data/processed/prediction_frame.csv", index=False)
