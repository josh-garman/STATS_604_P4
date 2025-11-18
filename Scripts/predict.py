#!/usr/bin/env python3
import warnings
import pandas as pd
from predict_functs import add_zone_predictions, peak_middle_idx_3hr, CrossBaseWeatherCats, predict

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

zone_order = [
    "AE",
    "AEPAPT",
    "AEPIMP",
    "AEPKPT",
    "AEPOPT",
    "AP",
    "BC",
    "CE",
    "DAY",
    "DEOK",
    "DOM",
    "DPLCO",
    "DUQ",
    "EASTON",
    "EKPC",
    "JC",
    "ME",
    "OE",
    "OVEC",
    "PAPWR",
    "PE",
    "PEPCO",
    "PLCO",
    "PN",
    "PS",
    "RECO",
    "SMECO",
    "UGI",
]

pred = pd.read_csv("Data/processed/prediction_frame.csv")

#if error occurs use prior month avg
try:
    pred = predict(pred, reg_model_dir = "Models/production", zones=None, 
                pred_col="mw_pred", reg_model_suffix="_ridge_pipeline.joblib", 
                class_model_dir = "Models/production")
except:
    # get date string for output
    pred["datetime_beginning_utc"] = pd.to_datetime(pred["datetime_beginning_utc"])
    pred_date = pd.to_datetime(pred["datetime_beginning_utc"].iloc[0]).date().isoformat()
    #outputs 
    out_vals = [pred_date]
    ph_list = []
    pd_list = []

    for zone in zone_order:
        z = pred[pred["load_area"] == zone].sort_values("Hour")

        # 24 hourly loads, rounded to nearest integer
        loads = z["mw_pred"].round().astype(int).tolist()
        out_vals.extend(loads)

        # peak hour + peak-day indicator (one per zone)
        ph_list.append(int(1))
        pd_list.append(int(22))

    # Append PH_1..PH_29 then PD_1..PD_29
    out_vals.extend(ph_list)
    out_vals.extend(pd_list)

    #Print exactly one CSV line, no other output
    print(", ".join(str(v) for v in out_vals))


df_pred = pred.copy()

# get date string for output
df_pred["datetime_beginning_utc"] = pd.to_datetime(df_pred["datetime_beginning_utc"])
pred_date = pd.to_datetime(df_pred["datetime_beginning_utc"].iloc[0]).date().isoformat()



#outputs 
out_vals = [pred_date]
ph_list = []
pd_list = []

for zone in zone_order:
    z = df_pred[df_pred["load_area"] == zone].sort_values("Hour")

    # 24 hourly loads, rounded to nearest integer
    loads = z["mw_pred"].round().astype(int).tolist()
    out_vals.extend(loads)

    # peak hour + peak-day indicator (one per zone)
    ph_list.append(int(z["peak_hour_idx"].iloc[0]))
    pd_list.append(int(z["is_pred_peak_day"].iloc[0]))

# Append PH_1..PH_29 then PD_1..PD_29
out_vals.extend(ph_list)
out_vals.extend(pd_list)

#Print exactly one CSV line, no other output
print(", ".join(str(v) for v in out_vals))
