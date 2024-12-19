# Weather Prediction using PyTorch

A multi-layer perceptron (MLP) model for predicting weather conditions using PyTorch. This project demonstrates the application of machine learning for forecasting weather metrics based on historical data.

## Features

- Implementation of an MLP in PyTorch.
- Preprocessing of weather data for model input.
- Training, validation, and testing pipelines.
- Visualization of model performance metrics.

## Installation

To run this project, you need to have Python 3.8+ installed along with the following dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Dataset

The project uses a weather dataset that includes historical weather features such as temperature, humidity, wind speed, etc. Ensure the dataset is in CSV format and includes the required features.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/weather-prediction-mlp.git
cd weather-prediction-mlp
```

### 2. Prepare the Dataset

Place your dataset in the `data/` directory and update the file path in the script.

### 3. Train the Model

Run the following command to train the MLP model:

```bash
python train.py
```

### 4. Evaluate the Model

After training, evaluate the model using:

```bash
python evaluate.py
```

### 5. Visualize Results

View the training and validation loss curves and other performance metrics:

```bash
python visualize.py
```

## Project Structure

```plaintext
weather-prediction-mlp/
|— data/                 # Dataset folder
|— models/               # Model definition
|— utils/                # Helper functions
|— train.py             # Training script
|— evaluate.py          # Evaluation script
|— visualize.py         # Visualization script
|— README.md            # Project documentation
```

## Model Architecture

The MLP consists of:

- Input layer matching the number of features in the dataset.
- One or more hidden layers with ReLU activation.
- Output layer for predicting weather metrics.

Example:

```python
import torch
import torch.nn as nn

class WeatherMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Results

- **Training Loss**: [value]
- **Validation Loss**: [value]
- **Test Accuracy**: [value]

Add visualizations of loss and predictions in your report for better understanding.

## Future Work

- Add hyperparameter tuning.
- Experiment with different model architectures.
- Include additional weather features for better accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- PyTorch Documentation
- Scikit-learn Library
- [Dataset Source](#)

---

Feel free to contribute to the project by submitting issues or pull requests!
