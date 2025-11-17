import pandas as pd

# 1) Load / set up data -----------------------------------------

# windows_df already exists, e.g.:
windows_df = pd.read_csv("Data/processed/daily_max_window_2024.csv")

# thresholds_df from your CSV
thresholds_df = pd.read_csv("Models/validation_and_training/zone_peak_day_thresholds.csv")

thresholds_df = thresholds_df.dropna(subset=["threshold"])

df = windows_df.merge(
    thresholds_df[["zone", "threshold"]],
    left_on="load_area",
    right_on="zone",
    how="inner"
)

# predicted label
df["is_pred_peak_in_window"] = (df["max_pred_mw"] >= df["threshold"]).astype(int)

# correctness
df["correct"] = (df["is_pred_peak_in_window"] == df["is_true_peak_in_window"])

# convenience booleans
df["true_peak"] = df["is_true_peak_in_window"]                    # y = 1
df["pred_peak"] = df["is_pred_peak_in_window"]                    # ŷ = 1
df["true_and_pred_peak"] = ((df["true_peak"] == 1) & (df["pred_peak"] == 1)).astype(int)

# aggregate core counts per load_area
agg = (
    df.groupby("load_area")
      .agg(
          n_rows=("correct", "size"),
          accuracy=("correct", "mean"),
          true_peaks=("true_peak", "sum"),                # TP + FN
          pred_peaks=("pred_peak", "sum"),                # TP + FP
          true_and_predicted=("true_and_pred_peak", "sum")# TP
      )
      .reset_index()
)

# derive confusion pieces
agg["TP"] = agg["true_and_predicted"]
agg["FN"] = agg["true_peaks"] - agg["TP"]
agg["FP"] = agg["pred_peaks"] - agg["TP"]

# percentages
agg["accuracy_pct"] = 100 * agg["accuracy"]
agg["peak_accuracy"] = agg["TP"] / agg["true_peaks"]              # recall on true peaks
agg["peak_accuracy_pct"] = 100 * agg["peak_accuracy"]

# custom loss: 4 * FN + 1 * FP
FN_COST = 4.0
FP_COST = 1.0

agg["loss"] = FN_COST * agg["FN"] + FP_COST * agg["FP"]
agg["loss_per_row"] = agg["loss"] / agg["n_rows"]

# if you just want the key columns:
result = agg[[
    "load_area",
    "accuracy_pct",
    "peak_accuracy_pct",
    "TP", "FN", "FP",
    "loss",
    "loss_per_row"
]].sort_values("loss_per_row")  # or sort however you want

print(result)

