# Google Stock Price Prediction

A deep learning project that predicts Google stock prices using a **stacked LSTM (Long Short-Term Memory)** neural network. The model is trained on historical opening prices and learns temporal patterns to forecast future stock values.

## Overview

This notebook implements a time-series forecasting pipeline using TensorFlow/Keras. It loads historical Google stock data, applies MinMax scaling, constructs 3-step sliding window sequences, trains a two-layer LSTM network with Dropout regularization, and visualizes real vs. predicted stock prices on a single chart.

## Dataset

- **File:** `Google_Stock_Price_Train.csv`
- **Column used:** `Open` (opening stock price)
- **Source:** Historical Google stock price data

## Project Structure

```
Google-Stock-Price-Prediction-main/
└── GoogleStockPrice.ipynb    # Main Jupyter Notebook
```

## Workflow

### 1. Data Loading & Scaling
- Load training CSV and extract the `Open` price column as a NumPy array
- Apply **MinMaxScaler** to normalize prices to the range `[0, 1]`

### 2. Sequence Construction
- Build sliding window sequences of **3 time steps**
- Each sample: 3 consecutive scaled prices → predict the next price
- Reshape input to `(samples, time_steps, features)` for LSTM compatibility

### 3. Model Architecture

```
Input (3 time steps × 1 feature)
  → LSTM(10 units, return_sequences=True)
  → Dropout(0.2)
  → LSTM(10 units, return_sequences=False)
  → Dropout(0.2)
  → Dense(1)    ← predicted next price
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Epochs:** 50
- **Batch Size:** 1

### 4. Prediction & Visualization
- Predict stock prices on the training set
- Inverse-transform scaled predictions back to original price range
- Plot **Real Google Stock Price** (red) vs **Predicted Google Stock Price** (blue)

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

## Usage

```bash
git clone https://github.com/your-username/Google-Stock-Price-Prediction.git
cd Google-Stock-Price-Prediction
jupyter notebook GoogleStockPrice.ipynb
```

Place `Google_Stock_Price_Train.csv` in the same directory as the notebook and update the file path in the load cell before running.

## Outputs

- Line chart comparing real vs. predicted Google stock opening prices over time

## Key Concepts

| Concept | Purpose |
|---|---|
| LSTM | Recurrent network layer designed to learn long-term temporal dependencies |
| MinMaxScaler | Normalizes prices to [0, 1] to stabilize LSTM training |
| Sliding Window (3 steps) | Creates time-series sequences so the model learns from recent history |
| Dropout (0.2) | Regularization to prevent overfitting on training sequences |
| Inverse Transform | Converts scaled predictions back to original stock price scale for visualization |
