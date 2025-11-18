#!/usr/bin/env python3
import numpy as np
import warnings
import pandas as pd
from pathlib import Path
from joblib import load as joblib_load
from sklearn.base import BaseEstimator, TransformerMixin
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Travis Dauwalter et al. (2022)
# “Coalition Stability in PJM: Exploring the Consequences of State Defection from the Wholesale Market.”
load_area_states = {
    "AE":   ["NJ"],                         # Atlantic Electric
    #"AEPAPT": ["TN", "VA", "WV"],            # AEP West (Appalachian)
    "AEPAPT": ["VA", "WV"],            # AEP West (Appalachian) #idt TN is useful 
    "AEPIMP": ["IN", "MI"],                  # AEP West (Indiana/Michigan)
    "AEPKPT": ["KY"],                        # AEP West (Kentucky Power)
    "AEPOPT": ["OH"],                        # AEP West (Ohio Power)

    "AP":     ["MD", "PA", "VA", "WV"],      # Allegheny Power Systems
    "BC":     ["MD"],                        # Baltimore Gas & Electric
    "CE":     ["IL"],                        # Commonwealth Edison
    "DAY":    ["OH"],                        # Dayton Power & Light
    "DEOK":   ["KY", "OH"],                  # Duke Energy Ohio & Kentucky

    # "DOM":   ["NC", "VA"],                   # Dominion
    "DOM":   ["VA"],                   # Dominion #idt NC is useful
    "DPLCO": ["DE", "MD", "VA"],             # Delmarva Power & Light
    "DUQ":   ["PA"],                         # Duquesne Light
    "EASTON":["MD"],                         # Delmarva (Easton sub-area)
    "EKPC":  ["KY"],                         # East Kentucky Power Cooperative

    "JC":    ["NJ"],                         # Jersey Central (JCP&L)
    "ME":    ["PA"],                         # MetEd
    "OE":    ["OH"],                         # ATSI – Ohio
    "OVEC":  ["OH"],                         # Ohio Valley Electric Corp
    "PAPWR": ["PA"],                         # ATSI – Pennsylvania

    "PE":    ["PA"],                         # PECO
    "PEPCO": ["DC", "MD"],                   # Potomac Electric Power
    "PLCO":  ["PA"],                         # PPL Electric Utilities
    "PN":    ["PA"],                         # PennElec (PENE)
    "PS":    ["NJ"],                         # PSEG
    "RECO":  ["NJ"],                         # Rockland Electric (NJ piece)
    "SMECO":["MD"],                          # Southern MD Electric Coop
    "UGI":   ["PA"],                         # UGI / PPL sub-area
}

#transformer class
class CrossBaseWeatherCats(BaseEstimator, TransformerMixin):
    """
    Input X is [base | weather | cats] from ColumnTransformer.
    Adds:
      - base × cats
      - base × weather
    """
    def __init__(self, n_base, n_weather):
        self.n_base = n_base
        self.n_weather = n_weather

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        n_base = self.n_base
        n_weather = self.n_weather
        n_cat = n_features - n_base - n_weather

        B = X[:, :n_base]                      # (n, n_base)
        W = X[:, n_base:n_base+n_weather]      # (n, n_weather)
        C = X[:, n_base+n_weather:]            # (n, n_cat)

        BC = (B[:, :, None] * C[:, None, :]).reshape(n_samples, -1)
        BW = (B[:, :, None] * W[:, None, :]).reshape(n_samples, -1)

        return np.hstack([B, W, C, BC, BW])

def peak_middle_idx_3hr(y_hat, window=3):
    """
    takes middle of max 3 rolling hour avg
    """
    y_hat = np.asarray(y_hat)
    n = y_hat.size

    # rolling_means[i] = mean(y_hat[i : i+window])
    rolling_means = np.convolve(y_hat, np.ones(window) / window, mode="valid")

    start_idx = int(rolling_means.argmax())   # start of best window
    middle_idx = start_idx + window // 2      # middle of that window
    max_mean = rolling_means[start_idx]

    return middle_idx


def add_zone_predictions(
    df,
    model_dir="Models",
    zones=None,
    pred_col="mw_pred",
    model_suffix="_ridge_pipeline.joblib",
):
    """
    model prediction hourly, outputs a df
    """
    model_dir = Path(model_dir)
    df_out = df.copy()

    if zones is None:
        zones = list(load_area_states.keys())

    for zone in zones:
        zone_mask = (df_out["load_area"] == zone)
        zone_idx = df_out.index[zone_mask]

        # if len(zone_idx) == 0:
        #     continue

        model_path = model_dir / f"{zone}{model_suffix}"
        model = joblib_load(model_path)

        # slice with the *original* index; do not sort, do not reindex
        df_zone = df_out.loc[zone_idx]

        y_hat = model.predict(df_zone)

        # # sanity check: one prediction per row
        # if len(y_hat) != len(zone_idx):
        #     raise RuntimeError(
        #         f"Prediction length mismatch for zone {zone}: "
        #         f"{len(y_hat)} preds vs {len(zone_idx)} rows."
        #     )

        # write preds back to exactly those rows
        df_out.loc[zone_idx, pred_col] = y_hat

    return df_out



def make_daily_scores(
    hourly_df,
    pred_col="mw_pred",
    window=3,
):
    """
    Collapse hourly predictions to daily scores per (load_area, date).

    For each (load_area, date):
      - max_pred_mw  = max over the day's predicted hours (used for thresholding)
      - peak_hour_idx = index (0-based within that day's sorted hours)
                        of the middle of the 3-hour window with max mean
    """
    df = hourly_df.copy()

    # ensure a date column exists
    if "date" not in df.columns:
        df["date"] = df["datetime_beginning_utc"].dt.date

    group_cols = ["load_area", "date"]

    def _summarize(group):
        g = group.sort_values("datetime_beginning_utc")
        y_pred = g[pred_col].values

        if len(y_pred) == 0:
            return pd.Series(
                {"max_pred_mw": np.nan, "peak_hour_idx": np.nan}
            )

        # daily score for thresholding
        max_pred_mw = float(y_pred.max())

        # find peak hour via 3-hour rolling mean
        if len(y_pred) < window:
            # fallback: just use the max hour index if we don't have enough hours
            peak_hour_idx = int(np.argmax(y_pred))
        else:
            peak_hour_idx = int(peak_middle_idx_3hr(y_pred, window=window))

        return pd.Series(
            {
                "max_pred_mw": max_pred_mw,
                "peak_hour_idx": peak_hour_idx,
            }
        )

    daily_df = (
        df.groupby(group_cols)
          .apply(_summarize)
          .reset_index()
    )

    return daily_df

def predict(df, reg_model_dir = "Models/production", zones=None, 
            pred_col="mw_pred", reg_model_suffix="_ridge_pipeline.joblib", 
            class_model_dir = None):
    pred_df = df.copy()
    pred_df = add_zone_predictions(
                pred_df,
                model_dir=reg_model_dir,
                zones=None,
                pred_col="mw_pred",
                model_suffix=reg_model_suffix,
            )
    daily_df = make_daily_scores(
                pred_df,
                pred_col="mw_pred",
                window=3,
            )
    if class_model_dir is not None:
        thresholds_df = pd.read_parquet(f"{class_model_dir}/zone_peak_day_thresholds.parquet")

        daily_df = daily_df.merge(
            thresholds_df[["zone", "threshold"]],
            left_on="load_area",
            right_on="zone",
            how="left"
        )

        daily_df["is_pred_peak_day"] = (daily_df["max_pred_mw"] >= daily_df["threshold"]).astype(int)

        daily_cols = [
            "load_area",
            "date",
            "max_pred_mw",
            "peak_hour_idx",
            "is_pred_peak_day",
        ]

        pred_df = pred_df.merge(
            daily_df[daily_cols],
            on=["load_area", "date"],
            how="left",
        )
    else:
        daily_cols = [
            "load_area",
            "date",
            "max_pred_mw",
            "peak_hour_idx",
        ]

        pred_df = pred_df.merge(
            daily_df[daily_cols],
            on=["load_area", "date"],
            how="left",
        )

    return pred_df
    