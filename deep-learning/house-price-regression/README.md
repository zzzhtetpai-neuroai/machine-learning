# House Price Regression

A neural network that predicts house sale prices in King County (Seattle area, USA) from features like size, rooms and location.

## The data

The **King County House Sales** dataset (`kc_house_data.csv`), with about 21,600 houses. The script downloads it automatically from the AI Crash Course GitHub repository, so you don't need the CSV file locally.

- **Target:** sale price
- **Features:** 17 numeric columns (bedrooms, bathrooms, living area, lot size, floors, waterfront, view, condition, grade, year built, location and others)

## How it works

1. Load the CSV and select the 17 feature columns and the price.
2. Split 80% for training and 20% for testing.
3. Scale both the features and the price to 0–1 with `MinMaxScaler`, so large prices don't dominate the training.
4. Train a feedforward network:

   ```
   Dense(64, ReLU) → Dense(16, ReLU) → Dense(1, ReLU)
   ```

   Loss: MSE, optimizer: Adam (lr 0.001), 100 epochs, batch size 32.
5. Convert the predictions back to real prices and report the **mean absolute percentage error** on the test set.

## Results

<!-- TODO: run the script and fill in the error -->
Average error on the test set: **…%** of the real price.

## Run it

```bash
pip install tensorflow keras scikit-learn pandas numpy
python house_price_regression.py
```

## What I learned

- The full supervised learning workflow: load, split, scale, train, evaluate.
- Why scaling the target matters in regression, and how to inverse-transform predictions back to real values.
- Reading a percentage error as a more intuitive metric than raw MSE.
