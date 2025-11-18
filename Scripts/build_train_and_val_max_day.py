#!/usr/bin/env python3
import pandas as pd
import warnings
from pathlib import Path
from predict_functs import add_zone_predictions, peak_middle_idx_3hr, CrossBaseWeatherCats, predict
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def build_sliding_df(daily_df, zone, n_days=120, window_size=10):
    daily_df = daily_df.copy()
    zone_daily = (
        daily_df[daily_df['load_area'] == zone]
        .sort_values('date')
        .reset_index(drop=True)
    )

    windows = []  # will hold info per window if you want

    n_days = len(zone_daily)

    # sliding windows: [0..9], [1..10], ..., [n_days-10..n_days-1]
    for start in range(0, n_days - window_size + 1):
        win = zone_daily.iloc[start:start+window_size].copy()
        win_id = f"{zone}_{start}"  # or any identifier you like

        # true top-2 days in THIS window
        win = win.sort_values('max_mw', ascending=False)
        top2_dates = set(win['date'].iloc[:2])

        # restore original order within window (optional, for clarity)
        win = win.sort_values('date')

        # add a label: 1 if this day is top-2 in this window, else 0
        win['is_true_peak_in_window'] = win['date'].isin(top2_dates).astype(int)

        # tag window id
        win['window_id'] = win_id

        windows.append(win)

    zone_windows_df = pd.concat(windows, ignore_index=True)
    return zone_windows_df

def build_all_windows(daily_df, n_days = 120, window_size=10, zones=None):
    """
    Build sliding windows for all zones and stack into a single dataframe.
    """
    if zones is None:
        zones = sorted(daily_df['load_area'].unique())

    all_windows = []

    for z in zones:
        z_win = build_sliding_df(daily_df, zone=z, n_days=n_days, window_size=window_size)
        # skip zones that don't have enough days
        if len(z_win) == 0:
            continue
        all_windows.append(z_win)

    all_windows_df = pd.concat(all_windows, ignore_index=True)
    return all_windows_df





model_dir = Path("Models/validation_and_training")

# full_data = pd.read_csv("Data/processed/full_data.csv")
full_data = pd.read_parquet("Data/processed/full_data.parquet")
pred_2023 = full_data[full_data["season_year"] == 2023]
pred_2024 = full_data[full_data["season_year"] == 2024]


#2023 preds 
# pred_2023 = add_zone_predictions(
#     pred_2023,
#     model_dir=model_dir,
#     zones=None,
#     pred_col="mw_pred",
#     model_suffix="_ridge_pipeline_val.joblib",
# )

pred_2023 = predict(pred_2023, reg_model_dir = "Models/validation_and_training", zones=None, 
            pred_col="mw_pred", reg_model_suffix="_ridge_pipeline_val.joblib")


# make sure it's a datetime
pred_2023["datetime_beginning_utc"] = pd.to_datetime(pred_2023["datetime_beginning_utc"])

# extract calendar day
pred_2023["date"] = pred_2023["datetime_beginning_utc"].dt.date

# roll up to daily, per load_area
daily_max_2023 = (
    pred_2023.groupby(["load_area", "date"], as_index=False)[["mw", "mw_pred"]]
      .max()
)

daily_max_2023 = daily_max_2023.rename(columns={
    "mw": "max_mw",
    "mw_pred": "max_pred_mw"
})

daily_max_window_2023 = build_all_windows(daily_max_2023, n_days = 120, window_size=10, zones=None)
# daily_max_2023.to_csv("Data/processed/daily_max_2023.csv", index=False)
# daily_max_window_2023.to_csv("Data/processed/daily_max_window_2023.csv", index=False)
daily_max_2023.to_parquet("Data/processed/daily_max_2023.parquet", index=False)
daily_max_window_2023.to_parquet("Data/processed/daily_max_window_2023.parquet", index=False)

#2024 preds 
pred_2024 = predict(pred_2024, reg_model_dir = "Models/validation_and_training", zones=None, 
            pred_col="mw_pred", reg_model_suffix="_ridge_pipeline_train.joblib")

# make sure it's a datetime
pred_2024["datetime_beginning_utc"] = pd.to_datetime(pred_2024["datetime_beginning_utc"])

# extract calendar day
pred_2024["date"] = pred_2024["datetime_beginning_utc"].dt.date

# roll up to daily, per load_area
daily_max_2024 = (
    pred_2024.groupby(["load_area", "date"], as_index=False)[["mw", "mw_pred","peak_hour_idx"]]
      .max()
)        

#get the max hour true val 
idx = pred_2024.groupby(["load_area", "date"])["mw"].idxmax()

hour_at_max = (
    pred_2024
      .loc[idx, ["load_area", "date", "Hour"]]
      .rename(columns={"Hour": "hour_at_max_mw"})
)

daily_max_2024 = daily_max_2024.merge(hour_at_max, on=["load_area", "date"], how="left")

daily_max_2024 = daily_max_2024.rename(columns={
    "mw": "max_mw",
    "mw_pred": "max_pred_mw",
    "hour_at_max_mw": "true_max_hr",
    "peak_hour_idx": "pred_max_hr"
})



daily_max_window_2024 = build_all_windows(daily_max_2024, n_days = 120, window_size=10, zones=None)

# daily_max_2024.to_csv("Data/processed/daily_max_2024.csv", index=False)
# daily_max_window_2024.to_csv("Data/processed/daily_max_window_2024.csv", index=False)
daily_max_2024.to_parquet("Data/processed/daily_max_2024.parquet", index=False)
daily_max_window_2024.to_parquet("Data/processed/daily_max_window_2024.parquet", index=False)
