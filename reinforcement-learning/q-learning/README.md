# Q-Learning Warehouse Routing

A warehouse robot learns the best route between locations using tabular Q-learning.

## The idea

The warehouse has 12 locations, **A to L**, connected like a small maze. The robot doesn't know the map in advance. It only gets rewards for allowed moves and a large reward (1000) for reaching the target. By exploring randomly and updating a Q-table, it learns which move is worth the most from every location, and then it just follows the highest values to get the route.

## How it works

- **States and actions:** 12 locations, and an action means "move to location j". A 12×12 reward matrix `R` marks which moves are allowed (1) and which are not (0).
- **Target:** the destination gets a reward of 1000 on itself, so the values "flow" toward it during training. Location G is set as the top priority location.
- **Training:** 1,000 iterations. Each one picks a random state, picks a random allowed action and applies the temporal difference update:

  ```
  TD = R[s, s'] + γ · max Q[s', :] − Q[s, s']
  Q[s, s'] += α · TD
  ```

  with `γ = 0.75` (discount) and `α = 0.9` (learning rate).
- **Route:** starting from the start location, repeatedly move to the location with the highest Q-value until the goal is reached.
- **Intermediate stop:** `best_route(start, via, end)` combines two routes, so the robot can be sent through a required location.

## Result

`best_route("E", "K", "G")` returns:

```
['E', 'I', 'J', 'K', 'L', 'H', 'G']
```

The robot goes from E through K and ends at G along the shortest path.

## Run it

```bash
pip install numpy
python q_learning.py
```

## What I learned

- How the Bellman equation and temporal difference updates turn sparse rewards into a full value table.
- What the discount factor γ and learning rate α actually control.
- Why tabular Q-learning only works for small state spaces. That limitation is what leads to Deep Q-Networks (see the other projects in this folder).
