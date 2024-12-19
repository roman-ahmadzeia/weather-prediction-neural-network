from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from flask_cors import CORS
import numpy as np

app = Flask(__name__)


CORS(app)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.output(x)
        return x


model = MLPClassifier(input_dim=6, output_dim=3, hidden_dim=200) 
model.load_state_dict(torch.load("model.pth"))
model.eval()


from sklearn.preprocessing import MinMaxScaler
def normalize(df):
  scaler = MinMaxScaler()
  scaler.fit(df.drop('rain', axis=1))
  scaled_features = scaler.transform(df.drop('rain', axis=1))
  scaled_df = pd.DataFrame(scaled_features, columns=df.drop('rain', axis=1).columns)
  scaled_df['rain'] = df['rain']
  return scaled_df
# Encodes the rain values into three distinct categories
def categorize_rain(rain_amount):
  if rain_amount == 0:
    return 0  # No rain if 0mm per hour
  elif rain_amount > 0 and rain_amount <= 2.5:
    return 1  # Light rain 0mm - 2.5mm per hour
  else:
    return 2  # Moderate/Heavy rain if > 2.5mm per hour

# Combines every two rows into a single maximized row
def max_df(df):
  num_groups = len(df) // 2
  max_df = df.groupby(df.index // 2).max()
  max_df = max_df[:num_groups]
  return max_df

import requests
import json
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define the API endpoint and parameters for Vancouver, BC
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 49.2497,
    "longitude": -123.1193,
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "dew_point_2m", "wind_speed_80m", "cloud_cover", "surface_pressure"],
    "forecast_days": 16
}


responses = openmeteo.weather_api(url, params=params)

response = responses[0]


hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_speed_80m = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_dew_point = hourly.Variables(3).ValuesAsNumpy()

hourly_data = {
    "temperature_2m": hourly_temperature_2m,
    "relative_humidity_2m": hourly_relative_humidity_2m,
    "dew_point_2m": hourly_dew_point,
    "surface_pressure": hourly_surface_pressure,
    "cloud_cover": hourly_cloud_cover,
    "wind_speed_80m": hourly_wind_speed_80m,
    "rain":hourly_rain
}

hourly_dataframe = pd.DataFrame(data=hourly_data)

# Take the maximum values from every 2 hours
num_groups = len(hourly_dataframe) // 2
max_df = hourly_dataframe.groupby(hourly_dataframe.index // 2).max()
max_df = max_df[:num_groups]
max_df = max_df.reset_index(drop=True)
hourly_dataframe = max_df



hourly_dataframe = normalize(hourly_dataframe)
hourly_dataframe['rain'] = hourly_dataframe['rain'].apply(categorize_rain)

targets = hourly_dataframe['rain'].to_list()
print(targets)
hourly_dataframe = hourly_dataframe.drop('rain', axis=1)
# Convert mean rain score to text
def mean_to_text(index):
    if index == 0:
        return "No Rain"
    # If it is mix of light and heavy, classify as heavy
    elif index > 0 and index <= 1:
        return "Light Rain"
    elif index > 1:
        return "Moderate/Heavy Rain"
    else:
        return "Unknown"

# Convert single rain score to text
def index_to_text(index):
    if index == 0:
        return "No Rain"
    elif index == 1:
        return "Light Rain"
    elif index == 2:
        return "Moderate/Heavy Rain"
    else:
        return "Unknown"

def generate_weather_summary(predictions, targets, days=16):
    # Predictions cover 2-hour intervals (12 per day)
    intervals_per_day = 12
    predictions = np.array(predictions)
    targets = np.array(targets)

    summary = []

    for day in range(days):
        day_start = day * intervals_per_day
        day_end = day_start + intervals_per_day

        # Extract predictions and targets for the current day
        daily_predictions = predictions[day_start:day_end]
        daily_targets = targets[day_start:day_end]

        # Compute statistics
        mean_pred = mean_to_text(np.mean(daily_predictions))
        min_pred = index_to_text(np.min(daily_predictions))
        max_pred = index_to_text(np.max(daily_predictions))
        actual_mean = mean_to_text(np.mean(daily_targets))

      
        summary.append({
            "Day": day + 1,
            "Mean Prediction": mean_pred,
            "Min Prediction": min_pred,
            "Max Prediction": max_pred,
            "Actual Mean": actual_mean
        })

    return summary

from flask import render_template
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    data = hourly_dataframe
    input_tensor = torch.tensor(data.values, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(torch.softmax(output, dim=1), dim=1).tolist()
        forecast = generate_weather_summary(predictions, targets)
    return jsonify({"prediction": forecast})

if __name__ == "__main__":
    app.run(debug=True)
