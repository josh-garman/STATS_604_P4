#!/usr/bin/env python3
import requests
import pandas as pd
import time
from datetime import timedelta

#hard coded coords from weather_coords.py 
#no need to keep pulling from api unless the cities grow legs 
STATE_CITY_COORDS = {
    "DE": [
        {"name": "Wilmington, Delaware", "lat": 39.7458, "lon": -75.5469},
        {"name": "Dover, Delaware",      "lat": 39.1582, "lon": -75.5244},
        {"name": "Newark, Delaware",     "lat": 39.6837, "lon": -75.7497},
        {"name": "Middletown, Delaware", "lat": 39.4496, "lon": -75.7163},
    ],
    "DC": [
        {"name": "Washington, District of Columbia", "lat": 38.89511, "lon": -77.03637},
    ],
    "IL": [
        {"name": "Chicago, Illinois", "lat": 41.85003, "lon": -87.65005},
    ],
    "IN": [
        {"name": "Fort Wayne, Indiana", "lat": 41.1306, "lon": -85.12886},
    ],
    "KY": [
        {"name": "Lexington, Kentucky", "lat": 37.98869, "lon": -84.47772},
    ],
    "MD": [
        {"name": "Baltimore, Maryland",     "lat": 39.29038, "lon": -76.61219},
        {"name": "Columbia, Maryland",      "lat": 39.2037,  "lon": -76.861},
        {"name": "Silver Spring, Maryland", "lat": 38.99067, "lon": -77.02609},
        {"name": "Frederick, Maryland",     "lat": 39.41427, "lon": -77.41054},
    ],
    "MI": [
        {"name": "Kalamazoo, Michigan", "lat": 42.29171, "lon": -85.58723},
    ],
    "NJ": [
        {"name": "Newark, New Jersey",       "lat": 40.73566, "lon": -74.17237},
        {"name": "Jersey City, New Jersey",  "lat": 40.72816, "lon": -74.07764},
        # {"name": "Paterson, New Jersey",     "lat": 40.91677, "lon": -74.17181},
        # {"name": "Elizabeth, New Jersey",    "lat": 40.66399, "lon": -74.2107},
    ],
    "OH": [
        {"name": "Columbus, Ohio",   "lat": 39.96118, "lon": -82.99879},
        {"name": "Cleveland, Ohio",  "lat": 41.4995,  "lon": -81.69541},
        {"name": "Cincinnati, Ohio", "lat": 39.12711, "lon": -84.51439},
        {"name": "Toledo, Ohio",     "lat": 41.66394, "lon": -83.55521},
    ],
    "PA": [
        {"name": "Philadelphia, Pennsylvania", "lat": 39.95238, "lon": -75.16362},
        {"name": "Pittsburgh, Pennsylvania",   "lat": 40.44062, "lon": -79.99589},
        {"name": "Allentown, Pennsylvania",    "lat": 40.60843, "lon": -75.49018},
        {"name": "Erie, Pennsylvania",         "lat": 42.12922, "lon": -80.08506},
    ],
    # "TN": [
    #     {"name": "Nashville, Tennessee",   "lat": 36.16589, "lon": -86.78444},
    #     {"name": "Memphis, Tennessee",     "lat": 35.14953, "lon": -90.04898},
    #     {"name": "Knoxville, Tennessee",   "lat": 35.96064, "lon": -83.92074},
    #     {"name": "Chattanooga, Tennessee", "lat": 35.04563, "lon": -85.30968},
    # ],
    "VA": [
        {"name": "Virginia Beach, Virginia", "lat": 36.85293, "lon": -75.97799},
        {"name": "Norfolk, Virginia",        "lat": 36.84681, "lon": -76.28522},
        {"name": "Chesapeake, Virginia",     "lat": 36.81904, "lon": -76.27494},
        {"name": "Richmond, Virginia",       "lat": 37.55376, "lon": -77.46026},
    ],
    "WV": [
        {"name": "Charleston, West Virginia",  "lat": 38.3498,  "lon": -81.6326},
        {"name": "Huntington, West Virginia",  "lat": 38.41925, "lon": -82.44515},
        {"name": "Morgantown, West Virginia",  "lat": 39.62953, "lon": -79.9559},
        {"name": "Parkersburg, West Virginia", "lat": 39.26674, "lon": -81.56151},
    ],
}


def get_daily_for_point_forecast(lat, lon, start_date, end_date, timezone="America/Detroit"):
    """
    Forecast daily temps for a single point (Open-Meteo forecast API).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        # same daily vars as historical, but from the forecast endpoint
        "daily": ["temperature_2m_min", "temperature_2m_max", "temperature_2m_mean"],
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")

def get_state_daily_temps_forecast(state_code, start_date, end_date):
    """
    Forecast daily temps for a state:
    simple average over all cities in STATE_CITY_COORDS[state_code].
    """
    cities = STATE_CITY_COORDS[state_code]
    dfs = []

    for city in cities:
        df_loc = get_daily_for_point_forecast(city["lat"], city["lon"], start_date, end_date)
        df_loc.columns = pd.MultiIndex.from_product([[city["name"]], df_loc.columns])
        dfs.append(df_loc)

    combined = pd.concat(dfs, axis=1)

    result = pd.DataFrame(index=combined.index)
    for var in ["temperature_2m_min", "temperature_2m_max", "temperature_2m_mean"]:
        result[var] = combined.xs(var, level=1, axis=1).mean(axis=1)

    result = result.rename(columns={
        "temperature_2m_min": "min",
        "temperature_2m_max": "max",
        "temperature_2m_mean": "mean",
    })
    return result


state_codes = [
    "DE", "DC", "IL", "IN", "KY",
    "MD", "MI", "NJ", "OH", "PA",
    # "TN",
    "VA", "WV",
]

# --- TOMORROW FORECAST PULL --- #
# "Tomorrow" in EST 
today_local = pd.Timestamp.now(tz="America/Detroit").normalize()
tomorrow = (today_local + pd.Timedelta(days=1)).date()
tomorrow_str = tomorrow.isoformat()  # 'YYYY-MM-DD'

# single-row index for tomorrow
dates = pd.date_range(start=tomorrow_str, end=tomorrow_str, freq="D")
df_forecast = pd.DataFrame(index=dates)
df_forecast.index.name = "date"

for state in state_codes:
    temp_vals = get_state_daily_temps_forecast(
        state,
        start_date=tomorrow_str,
        end_date=tomorrow_str,
    )
    time.sleep(1)

    df_forecast[f"min_{state}"]  = temp_vals["min"]
    df_forecast[f"max_{state}"]  = temp_vals["max"]
    df_forecast[f"mean_{state}"] = temp_vals["mean"]

df_forecast = df_forecast.reset_index().rename(columns={"index": "date"})

# df_forecast.to_csv("Data/intermediate/weather_tomorrow_forecast.csv", index=False)
df_forecast.to_parquet("Data/intermediate/weather_tomorrow_forecast.parquet", index=False)
