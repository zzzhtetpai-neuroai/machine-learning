# Classification and Regression

The three classic problem types every deep learning practitioner starts with: binary classification, multiclass classification and regression.

## Scripts

| File | Problem | Dataset | Output layer and loss |
|---|---|---|---|
| `imdb_binary_classification.py` | Is a movie review positive or negative? | IMDB, 50,000 reviews | 1 unit, sigmoid, binary crossentropy |
| `reuters_multiclass.py` | Which of 46 topics is a news article about? | Reuters newswires | 46 units, softmax, sparse categorical crossentropy |
| `boston_housing_regression.py` | What is a house's price? | Boston Housing, 506 samples | 1 unit, no activation, MSE |

All datasets are built into Keras and download automatically.

## How they work

- **Text to numbers:** reviews and articles are lists of word indices. They are **multi-hot encoded** into 10,000-dimensional vectors (1 if a word appears, 0 if not).
- **Validation split:** part of the training data is held out to watch for overfitting. Each script plots training and validation loss and accuracy.
- **Regression specifics:** features are normalized using the training data's mean and std only (never the test data, since that would be cheating). With so few samples, the script uses **K-fold cross-validation** (k = 4) to get a reliable validation score, then trains a final model.

## Run it

```bash
pip install tensorflow matplotlib numpy
python imdb_binary_classification.py
python reuters_multiclass.py      # reuses vectorize() from the IMDB script
python boston_housing_regression.py
```

## What I learned

- How the choice of last layer and loss follows directly from the type of problem.
- Reading training and validation curves to spot overfitting and pick the number of epochs.
- Why K-fold validation matters when the dataset is small.
- Normalizing features without leaking test data.

## Credits

Based on *Deep Learning with Python* by François Chollet (Manning).
