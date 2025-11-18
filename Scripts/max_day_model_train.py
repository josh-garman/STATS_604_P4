#!/usr/bin/env python3
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_zone_threshold(zone_windows_df, n_grid=25):
    # scores and labels
    scores = zone_windows_df['max_pred_mw'].values
    labels = zone_windows_df['is_true_peak_in_window'].values  # 0/1
    window_ids = zone_windows_df['window_id'].values

    # candidate thresholds: a grid over the range of scores
    # (you can also use unique scores, but this is usually enough)
    lo, hi = scores.min(), scores.max()
    thresholds = np.linspace(lo, hi, n_grid)

    best_T = None
    best_loss = np.inf

    for T in thresholds:
        # predicted peak-day indicator at this threshold
        pred = (scores >= T).astype(int)

        df_tmp = pd.DataFrame({
            'window_id': window_ids,
            'label': labels,
            'pred': pred,
        })

        # per-window FN and FP
        def window_loss(g):
            fn = ((g['label'] == 1) & (g['pred'] == 0)).sum()
            fp = ((g['label'] == 0) & (g['pred'] == 1)).sum()
            #return 4 * fn + 1 * fp
            return 4 * fn + 1 * fp

        loss_per_window = df_tmp.groupby('window_id').apply(window_loss)
        total_loss = loss_per_window.sum()

        if total_loss < best_loss:
            best_loss = total_loss
            best_T = T

    return best_T, best_loss


def train_zone_thresholds(daily_df, zones, save_dir, n_grid=25):
    if zones is None:
        zones = sorted(daily_df['load_area'].unique())

    records = []

    for zone in zones:
        #need to swap out with actual stuff.
        zone_windows_df = daily_df[daily_df["load_area"] == zone]
        # zone_windows_df = build_sliding_df(daily_df = daily_df, zone = zone, n_days = 120, window_size = 10)
        
        T_z, loss_z = compute_zone_threshold(zone_windows_df, n_grid=n_grid)
        # if zone is "RTO":
        #     print(T_Z)
        records.append({
            "zone": zone,
            "threshold": T_z,
            "loss": loss_z,
        })

    # records.append({
    #     "zone": zone,
    #     "threshold": T_z,
    #     "loss": loss_z,
    #     "n_windows": zone_windows_df['window_id'].nunique(),
    #     "n_rows": len(zone_windows_df),
    # })

    thresholds_df = pd.DataFrame(records)
    # save to disk
    # thresholds_df.to_csv(f"{save_dir}/zone_peak_day_thresholds.csv", index=False)
    thresholds_df.to_parquet(f"{save_dir}/zone_peak_day_thresholds.parquet", index=False)

# daily_df = pd.read_csv("Data/processed/daily_max_window_2023.csv")
daily_df = pd.read_parquet("Data/processed/daily_max_window_2023.parquet")
save_dir = "Models/validation_and_training"
train_zone_thresholds(daily_df = daily_df, zones = None, save_dir = save_dir)

daily_df = pd.read_parquet("Data/processed/daily_max_window_2024.parquet")
save_dir = "Models/production"
train_zone_thresholds(daily_df = daily_df, zones = None, save_dir = save_dir)


