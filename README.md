# Weather Prediction using PyTorch

A multi-layer perceptron (MLP) model for predicting rain forecase for the following weeks in Vancouver region using PyTorch.

![Alt text](/demo.png)

## Features

- Implementation of an MLP in PyTorch.
- Preprocessing of live weather data for model input.
- Training, validation, and testing of model.
- Visualization of model performance metrics.

## Installation

To run this project, you need to have Python 3.8+ installed along with the following dependencies:

```bash
pip install flask flask-cors torch numpy pandas scikit-learn requests requests-cache retry-requests openmeteo_requests

```

## Dataset

The model is trained on high-resolution hourly weather data obtained from OpenMeteo's API for Vancouver, British Columbia. This data provides detailed, granular insights into various meteorological variables, ensuring the model has access to precise and timely information. The dataset includes the following key parameters:

- **Temperature at 2 meters above ground**: Captures the air temperature near the surface, a critical metric for predicting weather patterns.
- **Relative Humidity at 2 meters**: Reflects the moisture content in the air, which can influence precipitation levels and rain classification.
- **Rainfall**: Provides the amount of rain in millimeters per hour, a central feature for categorizing rain into different intensities such as "No Rain," "Light Rain," and "Moderate/Heavy Rain."
- **Dew Point**: Indicates the temperature at which air becomes saturated with moisture, which helps predict fog and precipitation.
- **Wind Speed at 80 meters above ground**: Offers insights into wind behavior, a significant factor in weather dynamics.
- **Cloud Cover**: Measures the percentage of sky covered by clouds, influencing sunlight exposure and weather conditions.
- **Surface Pressure**: Reflects atmospheric pressure at the surface level, aiding in weather forecasting and storm prediction.

The use of high-resolution hourly data ensures the model captures subtle changes in weather conditions, enabling it to make accurate predictions. By training on data specific to Vancouver, BC, the model learns localized weather patterns, such as how certain combinations of variables typically lead to rain in this region.

### Data Preprocessing
Before training, the raw data undergoes preprocessing steps:
1. **Normalization**: To bring all features to a comparable scale using MinMaxScaler.
2. **Rain Categorization**: Converts raw rainfall values into discrete categories:
   - **No Rain**: 0 mm/h
   - **Light Rain**: >0 mm/h to ≤2.5 mm/h
   - **Moderate/Heavy Rain**: >2.5 mm/h
3. **Hourly Aggregation**: Every two consecutive hourly data points are combined into one by taking the maximum value, reducing noise and improving model robustness.

### Training Goal
The model is designed to predict rain categories based on meteorological inputs.

By leveraging OpenMeteo’s data and combining it with neural network techniques, this model offers weather predictions for Vancouver, BC.



Feel free to contribute to the project by submitting issues or pull requests!
