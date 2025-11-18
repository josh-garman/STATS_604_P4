#!/usr/bin/env python3
import warnings
import pandas as pd
import numpy as np
import joblib
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

            # base × cats  -> (n, n_base * n_cat)
            BC = (B[:, :, None] * C[:, None, :]).reshape(n_samples, -1)

            # base × weather -> (n, n_base * n_weather)
            BW = (B[:, :, None] * W[:, None, :]).reshape(n_samples, -1)

            # final: [B, W, C, BC, BW]
            return np.hstack([B, W, C, BC, BW])


def make_linear_pipeline(base_numeric, weather_cols, categorical_features, penalty="ridge", alpha=1.0, l1_ratio=0.5):
    """
    weather_cols: list of strings like ["min_DE", "max_DE", "mean_DE", "min_NJ", ...]
    penalty: "ols", "ridge", "lasso", "enet"
    """
  
    preprocess = ColumnTransformer(
        transformers=[
            ("base",    StandardScaler(),           base_numeric),
            ("weather", StandardScaler(),           weather_cols),
            ("cats",    OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ],
        remainder="drop"
    )

    #model 
    if penalty == "ols":
        estimator = LinearRegression()
    elif penalty == "ridge":
        estimator = Ridge(alpha=alpha)
    elif penalty == "lasso":
        estimator = Lasso(alpha=alpha, max_iter=10000)
    elif penalty == "enet":
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    else:
        raise ValueError(f"Unknown penalty {penalty}")

    n_base = len(base_numeric)
    n_weather = len(weather_cols)

    
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("cross", CrossBaseWeatherCats(n_base=n_base, n_weather=n_weather)),
        ("model", estimator),
    ])

    # Also return the columns the pipeline expects
    x_cols = base_numeric + weather_cols + categorical_features
    return pipe, x_cols




def tune_ridge_on_val(
    df_train,
    df_val,
    base_numeric,
    weather_cols,
    categorical_features,
    alphas,
    penalty="ridge",
    model_path=None,
    df_test=None,
    zone=None
):
    """
    df_train, df_val: dataframes already subset exactly how you want
                      (e.g. winters only, specific years, single zone)

    base_numeric:     list of names of base numeric features
    weather_cols:     list of names of weather features
    categorical_features: list of cat feature names, e.g. ["Day_of_Week", "Hour"]
    alphas:           list of ridge alphas to try, e.g. [0.1, 1, 5, 10, 50, 100]
    """

    df_train = df_train.copy()
    df_val   = df_val.copy()
    
    x_cols = base_numeric + list(weather_cols) + categorical_features

    X_train = df_train[x_cols]
    y_train = df_train["mw"].values

    X_val = df_val[x_cols]
    y_val = df_val["mw"].values

    best_alpha = None
    best_rmse  = float("inf")
    results    = []

    for a in alphas:
        pipe, _ = make_linear_pipeline(
            base_numeric=base_numeric,
            weather_cols=weather_cols,
            categorical_features=categorical_features,
            penalty=penalty,
            alpha=a,
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred)**(1/2)
        mae  = mean_absolute_error(y_val, y_pred)
        r2   = r2_score(y_val, y_pred)

        results.append({"alpha": a, "rmse": rmse, "mae": mae, "r2": r2})

        if rmse < best_rmse:
            best_rmse  = rmse
            best_alpha = a

    df_full = pd.concat([df_train, df_val], axis=0)

    X_full = df_full[x_cols]
    y_full = df_full["mw"].values

    final_pipe, _ = make_linear_pipeline(
        base_numeric=base_numeric,
        weather_cols=weather_cols,
        categorical_features=categorical_features,
        penalty=penalty,
        alpha=best_alpha,
    )

    final_pipe.fit(X_full, y_full)
    if model_path is not None:
        dump(final_pipe, model_path)

    if df_test is not None:
        X_test = df_test[x_cols]
        y_test = df_test["mw"].values
        y_pred_test = final_pipe.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred_test)**(1/2)
        mae  = mean_absolute_error(y_test, y_pred_test)
        r2   = r2_score(y_test, y_pred_test)

        #saving for visualization
        if zone in ["OVEC", "AP"]:
            # put into a DataFrame
            df_test_val = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred_test,
            })

            # save
            df_test_val.to_csv(f"results/{zone}_hourly_pred_2024.csv", index=False)

        return [rmse, mae, r2]

    return best_alpha, results


def split_winter_train_val_test(df, val_season):
    "splits into test, train, val based on season year, must validate on future years due to leakage"
    # keep only winter months 8–12 and 1–2
    train_mask = df["season_year"] < val_season
    val_mask   = df["season_year"] == val_season
    test_mask = (df["season_year"] > val_season) & (df["season_year"] < 2025)

    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()
    df_test  = df[test_mask].copy()

    return df_train, df_val, df_test



def get_weather_cols_for_zone(zone, load_area_states):
    "gets weather features for each model"
    states = load_area_states[zone]
    weather_cols = []

    for st in states:
        # full list of expected columns for this state
        
        expected = [
                f"min_{st}", f"max_{st}", f"mean_{st}",
                f"min_{st}_sq", f"max_{st}_sq", f"mean_{st}_sq",
                f"min_{st}_cu", f"max_{st}_cu", f"mean_{st}_cu",
                f"mean_{st}_below_60", f"mean_{st}_above_60",
                f"mean_{st}_below_60_sq", f"mean_{st}_above_60_sq",
        ]        

        weather_cols.extend(expected)

    #just in case 
    return sorted(weather_cols)


#FITTING LOOP


#same for all zones
base_numeric = [
    "mw_lag_1y", "prev_month_mean_mw",
    "mw_lag_1y_sq", "prev_month_mean_mw_sq",
    "mw_lag_1y_cu", "prev_month_mean_mw_cu",
]


categorical_features = ["Day_of_Week", "Hour", "is_weekend", "is_holiday"]

#tuning parmas
alphas = [0.1, 1, 5, 10, 50, 100, 500, 1000]

#df
# df = pd.read_csv("Data/processed/full_data.csv")
df = pd.read_parquet("Data/processed/full_data.parquet")
df["Day_of_Week"] = df["Day_of_Week"].astype("category")
df["Hour"] = df["Hour"].astype("category")

test_results = [] 

for zone in load_area_states:
    df_zone = df[df["load_area"] == zone].copy()

    #weather cols 
    weather_cols = get_weather_cols_for_zone(zone, load_area_states)
    
    #validation and training models:
    #train on 2020-2021
    #validation on 2022
    #predict on 2023
    #used for validating maximum for week model 
    df_train, df_val, df_test = split_winter_train_val_test(
        df_zone,
        val_season=2022
    )

    model_path = f"Models/validation_and_training/{zone}_ridge_pipeline_val.joblib"
    
    results = tune_ridge_on_val(
        df_train=df_train,
        df_val=df_val,
        base_numeric=base_numeric,
        weather_cols=weather_cols,
        categorical_features=categorical_features,
        alphas=alphas,
        penalty="ridge",
        model_path=model_path,
        df_test=None
    )


    #train on 2020-2022
    #validation on 2023
    #predict on 2024
    #used for training maximum for week model 
    df_train, df_val, df_test = split_winter_train_val_test(
        df_zone,
        val_season=2023
    )
    
    model_path = f"Models/validation_and_training/{zone}_ridge_pipeline_train.joblib"
    
    results = tune_ridge_on_val(
        df_train=df_train,
        df_val=df_val,
        base_numeric=base_numeric,
        weather_cols=weather_cols,
        categorical_features=categorical_features,
        alphas=alphas,
        penalty="ridge",
        model_path=model_path,
        df_test=df_test,
        zone=zone
    )

    test_results.append({
        "zone": zone,
        "rmse": results[0],
        "mae": results[1],
        "r2": results[2],
    })

    #production model 
    #train on 2020-2023
    #validation on 2024
    #used for training maximum for week model 
    df_train, df_val, df_test = split_winter_train_val_test(
        df_zone,
        val_season=2024
    )

    model_path = f"Models/production/{zone}_ridge_pipeline.joblib"
    
    results = tune_ridge_on_val(
        df_train=df_train,
        df_val=df_val,
        base_numeric=base_numeric,
        weather_cols=weather_cols,
        categorical_features=categorical_features,
        alphas=alphas,
        penalty="ridge",
        model_path=model_path,
        df_test=None
    )
    

testing_results = pd.DataFrame(test_results)
testing_results.to_csv("results/testing_results_hourly_pred_2024.csv", index=False)