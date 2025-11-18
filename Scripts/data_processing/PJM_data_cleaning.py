#!/usr/bin/env python3
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#PJM_files
files = [
    "Data/raw/hrl_load_metered_2018.csv",
    "Data/raw/hrl_load_metered_2019.csv",
    "Data/raw/hrl_load_metered_2020.csv",
    "Data/raw/hrl_load_metered_2021.csv",
    "Data/raw/hrl_load_metered_2022.csv",
    "Data/raw/hrl_load_metered_2023.csv",
    "Data/raw/hrl_load_metered_2024.csv",
    "Data/raw/hrl_load_metered_2025.csv",
]

#dates
date_cols = ["datetime_beginning_utc", "datetime_beginning_ept"]
dt_format = "%m/%d/%Y %I:%M:%S %p"

#load into one df
df = pd.concat(
    (pd.read_csv(
            f,
            parse_dates=date_cols,
            date_format=dt_format,)
        for f in files
    ),
    ignore_index=True
)

#dropping cols 
df = df.drop(columns=["datetime_beginning_ept", "nerc_region", "mkt_region","zone","is_verified"])

#rolling AECO and VMUE into AE
df['load_area'] = df['load_area'].replace({
    'AECO': 'AE',
    'VMEU': 'AE',   
})

group_cols = [
    'datetime_beginning_utc',
    'load_area', 
]

df_rolled = (
    df
    .groupby(group_cols, as_index=False)
    .agg({'mw': 'sum'})
)
df = df_rolled

#dates 
df = df.sort_values(["load_area", "datetime_beginning_utc"])
df["Year"] = df["datetime_beginning_utc"].dt.year
df["Month"] = df["datetime_beginning_utc"].dt.month
df["Day"] = df["datetime_beginning_utc"].dt.day
df["Hour"] = df["datetime_beginning_utc"].dt.hour # Time of day
df["Day_of_Week"] = df["datetime_beginning_utc"].dt.dayofweek #Day of Week

#feature engineering time covs  

##############
# 1-year lag #
##############
df["datetime_minus_1y"] = df["datetime_beginning_utc"] - pd.DateOffset(years=1)

#lag lookback 
lag_lookup = (
    df[["load_area", "datetime_beginning_utc", "mw"]]
      .rename(columns={
          "datetime_beginning_utc": "datetime_minus_1y",
          "mw": "mw_lag_1y",
      })
)

# marge 
df = df.merge(
    lag_lookup,
    on=["load_area", "datetime_minus_1y"],
    how="left"
)

#don't need anymore
df = df.drop(columns=["datetime_minus_1y"])

###############
# 1 month avg #
###############

#Year month 
df["year_month"] = df["datetime_beginning_utc"].dt.to_period("M")

#Monthly Avg 
monthly = (
    df.groupby(["load_area", "Hour", "year_month"], as_index=False)["mw"]
      .mean()
      .rename(columns={"mw": "month_mean_mw"})
)

#prior month 
monthly["prev_month_mean_mw"] = (
    monthly
    .groupby(["load_area", "Hour"])["month_mean_mw"]
    .shift(1)
)

#merge
df = df.merge(
    monthly[["load_area", "Hour", "year_month", "prev_month_mean_mw"]],
    on=["load_area", "Hour", "year_month"],
    how="left"
)

#don't need anymore 
df = df.drop(columns=["year_month"])

#subsetting for years we want
# #2019 special case 
# years_keep = [2020, 2021, 2022, 2023, 2024, 2025]
# months_keep = [1, 2, 10, 11, 12] 

# df_2019 = df[(df["Year"] == 2019) & (df["Month"].isin([10, 11, 12]))]
# df_other_years = df[
#     df["Year"].isin(years_keep)
#     & df["Month"].isin(months_keep)
# ]

# df = pd.concat([df_2019, df_other_years], ignore_index=True)

#2020 special case 
years_keep = [2021, 2022, 2023, 2024, 2025]
months_keep = [1, 2, 10, 11, 12] 

df_2020 = df[(df["Year"] == 2020) & (df["Month"].isin([10, 11, 12]))]
df_other_years = df[
    df["Year"].isin(years_keep)
    & df["Month"].isin(months_keep)
]

df = pd.concat([df_2020, df_other_years], ignore_index=True)

# df.to_csv("Data/intermediate/PJM_intermediate.csv", index=False)
df.to_parquet("Data/intermediate/PJM_intermediate.parquet", index=False)



