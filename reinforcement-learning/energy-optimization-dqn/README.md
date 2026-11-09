# Energy Optimization DQN

A Deep Q-Network that controls a data center server's cooling and heating to save energy compared with a standard built-in cooling system.

## The idea

A server's temperature rises and falls with the outside temperature, the number of users and the data traffic. A normal cooling system reacts only after the temperature leaves the safe range (18 to 24 °C). The AI instead learns to adjust the temperature in advance, spending less energy overall. The idea is similar to how DeepMind cut Google's data center cooling costs with ML.

## How it works

**Environment (`energy_model_environment.py`):**

- Server temperature = outside temperature (by month) + 1.25 × users + 1.25 × data rate
- Users (10 to 100) and data rate (20 to 300) drift randomly every minute
- One step = one minute of simulated time

**State (3 inputs, scaled 0 to 1):** server temperature, number of users and data transmission rate.

**Actions (5):** cool by 3.0 °C, cool by 1.5 °C, do nothing, heat by 1.5 °C or heat by 3.0 °C.

**Reward:** energy saved compared with the built-in system, `0.001 × (energy_no_ai − energy_ai)`. During training, the episode ends if the temperature goes outside −20 to 80 °C.

**Brain (`energy_brain.py`, Keras):**

```
Input(3) → Dense(64, sigmoid) → Dropout(0.1) → Dense(32, sigmoid) → Dropout(0.1) → Dense(5)
```

**Training (`energy_train.py`):** up to 100 epochs, each starting in a random month and running up to 5 simulated months. ε = 0.3, replay memory 3,000, batch 512, learning rate 0.00001, with **early stopping** (patience 10). Saves to `model.keras`.

**Testing (`energy_test.py`):** runs the trained model for a full simulated year (518,400 minutes) and compares total energy with and without AI.

## Files

| File | Purpose |
|---|---|
| `energy_model_environment.py` | Server and temperature simulation |
| `energy_brain.py` | Neural network |
| `energy_DQN.py` | Experience replay and Q-learning targets |
| `energy_train.py` | Training loop with early stopping |
| `energy_test.py` | One-year evaluation |
| `model.keras` | Trained model |

## Results

<!-- TODO: run energy_test.py and fill in the numbers -->
| | Total energy |
|---|---|
| Without AI | … |
| With AI | … |
| **Energy saved** | **…%** |

## Run it

```bash
pip install tensorflow keras numpy
python energy_train.py
python energy_test.py
```

## What I learned

- Turning a real-world control problem into states, actions and rewards.
- Defining the reward as an improvement over a baseline instead of an absolute value.
- Using early stopping in reinforcement learning, not only in supervised learning.
- Fixing random seeds (`PYTHONHASHSEED`, NumPy and `random`) for reproducible experiments.

## Credits

Based on the energy optimization project in *AI Crash Course* by Hadelin de Ponteves (Packt, 2019).
