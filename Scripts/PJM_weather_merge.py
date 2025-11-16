import pandas as pd


PJM = pd.read_csv("Data/intermediate/PJM_intermediate.csv")
PJM["datetime_beginning_utc"] = pd.to_datetime(PJM["datetime_beginning_utc"])
PJM["date"] = PJM["datetime_beginning_utc"].dt.floor("D")

weather = pd.read_csv("Data/intermediate/weather_intermediate.csv")
weather["date"] = pd.to_datetime(weather["date"])

merged = PJM.merge(weather, on="date", how="left")

merged.to_csv("Data/processed/full_data.csv", index=False)


# print(merged.head())

