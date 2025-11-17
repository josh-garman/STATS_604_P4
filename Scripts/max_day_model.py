
import numpy as np
import pandas as pd


# def build_sliding_df(daily_df, zone, n_days, window_size):
#     daily_df = daily_df.copy()
#     zone_daily = (
#         daily_df[daily_df['load_area'] == zone]
#         .sort_values('date')
#         .reset_index(drop=True)
#     )

#     window_size = 10
#     windows = []  # will hold info per window if you want

#     n_days = len(zone_daily)

#     # sliding windows: [0..9], [1..10], ..., [n_days-10..n_days-1]
#     for start in range(0, n_days - window_size + 1):
#         win = zone_daily.iloc[start:start+window_size].copy()
#         win_id = f"{zone}_{start}"  # or any identifier you like

#         # true top-2 days in THIS window
#         win = win.sort_values('max_mw', ascending=False)
#         top2_dates = set(win['date'].iloc[:2])

#         # restore original order within window (optional, for clarity)
#         win = win.sort_values('date')

#         # add a label: 1 if this day is top-2 in this window, else 0
#         win['is_true_peak_in_window'] = win['date'].isin(top2_dates).astype(int)

#         # tag window id
#         win['window_id'] = win_id

#         windows.append(win)

#     zone_windows_df = pd.concat(windows, ignore_index=True)
#     return zone_windows_df

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
    thresholds_df.to_csv(f"{save_dir}/zone_peak_day_thresholds.csv", index=False)

daily_df = pd.read_csv("Data/processed/daily_max_window_2023.csv")
save_dir = "Models/validation_and_training"
train_zone_thresholds(daily_df = daily_df, zones = None, save_dir = save_dir)

daily_df = pd.read_csv("Data/processed/daily_max_window_2024.csv")
save_dir = "Models/production"
train_zone_thresholds(daily_df = daily_df, zones = None, save_dir = save_dir)


