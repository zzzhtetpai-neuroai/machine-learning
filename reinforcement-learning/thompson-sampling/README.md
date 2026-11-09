# Thompson Sampling

Picking the best option when you don't know in advance which one is best, using the multi-armed bandit problem.

## The idea

Imagine several slot machines, each paying out with a different hidden probability. You want to find the best one without wasting too many plays on bad ones. Thompson sampling keeps a Beta distribution of beliefs for each machine, draws a random sample from each one every round and plays the machine with the highest sample. Machines that win often get sharper, higher beliefs and get picked more, while uncertain machines still get explored now and then.

## Scripts

| File | What it does |
|---|---|
| `thompson_sampling.py` | 5 slot machines with conversion rates 15%, 4%, 13%, 11% and 5%. Runs 10,000 rounds and reports how often each machine was chosen. |
| `thompson_sampling_plot.py` | Business version: 9 marketing strategies (conversion rates from 1% to 20%) shown to 10,000 customers. Compares Thompson sampling with random selection and plots a histogram of the strategies chosen. |

## How it works

1. Simulate the ground truth: for every round and every option, decide in advance whether it would convert, based on its rate.
2. Each round, draw `beta(wins + 1, losses + 1)` for every option and pick the highest draw.
3. Update that option's win or loss count with the result.
4. (Plot script) Run random selection side by side and compute the relative return:
   `(reward_thompson - reward_random) / reward_random * 100`.

## Results

From my own runs (the results change slightly each run because there is no fixed seed):

- `thompson_sampling.py` correctly identifies **Machine 1** (15%) as the best choice.
- `thompson_sampling_plot.py` gives a **relative return of about 80 to 100%** over random selection, so Thompson sampling gets roughly twice as many conversions. The histogram shows it concentrating on strategy 7 (20%, the best one).

## Run it

```bash
pip install numpy matplotlib
python thompson_sampling.py
python thompson_sampling_plot.py
```

## What I learned

- The exploration vs exploitation trade-off and why always picking the current best option can lock you into a bad choice.
- How a Beta distribution models uncertainty about a success rate.
- How to measure an algorithm against a baseline (random selection) instead of judging it alone.
