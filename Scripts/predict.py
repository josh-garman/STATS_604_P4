import pandas as pd
from predict_functs import add_zone_predictions, peak_middle_idx_3hr, CrossBaseWeatherCats, predict

pred = pd.read_csv("Data/processed/prediction_frame.csv")

pred = predict(pred, reg_model_dir = "Models/production", zones=None, 
            pred_col="mw_pred", reg_model_suffix="_ridge_pipeline.joblib", 
            class_model_dir = "Models/production")

print(pred.shape)