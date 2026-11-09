# Time Series Forecasting

Predicting the temperature 24 hours ahead from weather sensor data, using recurrent neural networks (RNNs).

## The data

The **Jena Climate** dataset: 14 weather measurements (temperature, pressure, humidity, wind and more) recorded every 10 minutes from 2009 to 2016.

- Split in time order: 50% train, 25% validation, 25% test (no shuffling across time, so the model never sees the future).
- Normalized with the training data's mean and std.
- Each sample covers the last **5 days** (120 timesteps, one per hour), and the target is the temperature **24 hours later**.

## Scripts

| File | Model |
|---|---|
| `temperature_lstm.py` | LSTM (32 units) with recurrent dropout and dropout |
| `temperature_stacked_gru.py` | Two stacked GRU layers with recurrent dropout |
| `temperature_bidirectional_lstm.py` | Bidirectional LSTM (reads the sequence forwards and backwards) |

All models are trained with MSE loss and report **MAE** (mean absolute error). The inputs are normalized, but the target is the real temperature, so the MAE is in °C.

## Data

Download `jena_climate_2009_2016.csv` (from the [Keras dataset link](https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip)) and put it in an `RNN/` folder next to the scripts.

## Run it

```bash
pip install tensorflow numpy matplotlib
python temperature_lstm.py
python temperature_stacked_gru.py
python temperature_bidirectional_lstm.py
```

## What I learned

- How to turn a long time series into training samples with `timeseries_dataset_from_array`.
- Why time series must be split by time, not shuffled randomly.
- LSTM vs GRU, recurrent dropout, stacking layers and bidirectional RNNs.
- Why bidirectional RNNs help with text but not much with weather: for forecasting, recent data matters most, and reading backwards adds little.
