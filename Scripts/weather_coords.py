import requests
import pandas as pd 

STATE_CITIES = {
    "DE": [
        {"city": "Wilmington", "state_name": "Delaware"},
        {"city": "Dover",      "state_name": "Delaware"},
        {"city": "Newark",     "state_name": "Delaware"},
        {"city": "Middletown", "state_name": "Delaware"},
    ],
    "DC": [
        {"city": "Washington", "state_name": "District of Columbia"},
    ],
    "IL": [
        {"city": "Chicago", "state_name": "Illinois"},
    ],
    "IN": [
        {"city": "Fort Wayne", "state_name": "Indiana"},
    ],
    "KY": [
        {"city": "Lexington", "state_name": "Kentucky"},
    ],
    "MD": [
        {"city": "Baltimore",      "state_name": "Maryland"},
        {"city": "Columbia",       "state_name": "Maryland"},
        {"city": "Silver Spring",  "state_name": "Maryland"},
        {"city": "Frederick",      "state_name": "Maryland"},
    ],
    "MI": [
        {"city": "Kalamazoo",     "state_name": "Michigan"},
    ],
    "NJ": [
        {"city": "Newark",       "state_name": "New Jersey"},
        {"city": "Jersey City",  "state_name": "New Jersey"},
        {"city": "Paterson",     "state_name": "New Jersey"},
        {"city": "Elizabeth",    "state_name": "New Jersey"},
    ],
    "OH": [
        {"city": "Columbus",   "state_name": "Ohio"},
        {"city": "Cleveland",  "state_name": "Ohio"},
        {"city": "Cincinnati", "state_name": "Ohio"},
        {"city": "Toledo",     "state_name": "Ohio"},
    ],
    "PA": [
        {"city": "Philadelphia", "state_name": "Pennsylvania"},
        {"city": "Pittsburgh",   "state_name": "Pennsylvania"},
        {"city": "Allentown",    "state_name": "Pennsylvania"},
        {"city": "Erie",         "state_name": "Pennsylvania"},
    ],
    "TN": [
        {"city": "Nashville",   "state_name": "Tennessee"},
        {"city": "Memphis",     "state_name": "Tennessee"},
        {"city": "Knoxville",   "state_name": "Tennessee"},
        {"city": "Chattanooga", "state_name": "Tennessee"},
    ],
    "VA": [
        {"city": "Virginia Beach", "state_name": "Virginia"},
        {"city": "Norfolk",        "state_name": "Virginia"},
        {"city": "Chesapeake",     "state_name": "Virginia"},
        {"city": "Richmond",       "state_name": "Virginia"},
    ],
    "WV": [
        {"city": "Charleston",  "state_name": "West Virginia"},
        {"city": "Huntington",  "state_name": "West Virginia"},
        {"city": "Morgantown",  "state_name": "West Virginia"},
        {"city": "Parkersburg", "state_name": "West Virginia"},
    ],
}



#get geeocode city 
def geocode_city(city_name, state_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json",
        "country": "US",
        "admin1": state_name,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    j = r.json()

    if "results" not in j or not j["results"]:
        raise ValueError(f"No geocoding result for {city_name}, {state_name}. Full response: {j}")

    res = j["results"][0]
    return res["latitude"], res["longitude"]

def build_state_city_coords(state_cities: dict) -> dict:
    """
    Convert STATE_CITIES dict of
        state_code -> list of {"city": ..., "state_name": ...}
    into
        state_code -> list of {"name": ..., "lat": ..., "lon": ...}
    """
    out = {}
    for state_code, cities in state_cities.items():
        out[state_code] = []
        for entry in cities:
            city_name = entry["city"]
            state_name = entry["state_name"]
            lat, lon = geocode_city(city_name, state_name)
            out[state_code].append({
                "name": f"{city_name}, {state_name}",
                "lat": lat,
                "lon": lon,
            })
    return out


STATE_CITY_COORDS = build_state_city_coords(STATE_CITIES)

#manual fixes
# Fix Delaware
for c in STATE_CITY_COORDS["DE"]:
    if c["name"].startswith("Wilmington"):
        c["lat"], c["lon"] = 39.7458, -75.5469
    elif c["name"].startswith("Dover"):
        c["lat"], c["lon"] = 39.1582, -75.5244
    elif c["name"].startswith("Newark"):
        c["lat"], c["lon"] = 39.6837, -75.7497
    elif c["name"].startswith("Middletown"):
        c["lat"], c["lon"] = 39.4496, -75.7163

# Fix Columbia, MD
for c in STATE_CITY_COORDS["MD"]:
    if c["name"].startswith("Columbia"):
        c["lat"], c["lon"] = 39.2037, -76.8610

# Fix Charleston, WV
for c in STATE_CITY_COORDS["WV"]:
    if c["name"].startswith("Charleston"):
        c["lat"], c["lon"] = 38.3498, -81.6326


print(STATE_CITY_COORDS)
# print(STATE_CITY_COORDS)

# #daily location temps
# def get_daily_for_point(lat, lon, start_date, end_date, timezone="America/Detroit"):
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": start_date,
#         "end_date": end_date,
#         "daily": ["temperature_2m_min", "temperature_2m_max", "temperature_2m_mean"],
#         "timezone": timezone,
#     }
#     r = requests.get(url, params=params)
#     r.raise_for_status()
#     j = r.json()
#     df = pd.DataFrame(j["daily"])
#     df["time"] = pd.to_datetime(df["time"])
#     return df.set_index("time")
