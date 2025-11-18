#!/usr/bin/env python3
import pandas as pd

df = pd.read_parquet("Data/processed/daily_max_2024.parquet")

#can be within +-1 
df["hr_diff"] = (df["pred_max_hr"] - df["true_max_hr"]).abs()

df["correct"] = df["hr_diff"] <= 1


load_area_summary = (
    df.groupby("load_area", as_index=False)["correct"]
      .mean()
      .rename(columns={"correct": "acc"})
)

# if you actually want percentages (0–100) instead of proportions (0–1):
load_area_summary["acc"] *= 100

load_area_summary.to_csv("results/hourly_max_pred_2024.csv")