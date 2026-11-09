# PyTorch Fundamentals

Learning PyTorch from the ground up: writing the training loop myself, then wrapping it into a reusable trainer class. Each script builds on the one before.

## Scripts

| File | What it adds |
|---|---|
| `training_loop.py` | A linear regression model (`y = 1 + 2x + noise`) trained with a hand-written training loop: train step, validation step and mini-batches with `DataLoader`. The learned weights should come out close to b = 1 and w = 2 |
| `train_with_validation.py` | Train/validation split with `random_split`, validation loss every epoch, **TensorBoard** logging and saving a **checkpoint** (model, optimizer and loss history) |
| `resume_training.py` | Loads the checkpoint and **continues training** where it stopped, and logs the model graph to TensorBoard |
| `step_by_step.py` | `StepByStep`: a reusable trainer class that bundles everything above (device handling, loaders, training, validation, TensorBoard, checkpoints, prediction and loss plots) |
| `train_with_class.py` | The same regression, now trained in a few lines using `StepByStep` |
| `moons_classification.py` | **Binary classification** on the scikit-learn "two moons" dataset with `BCEWithLogitsLoss`, using `StepByStep` |

## The training loop pattern

```
for each epoch:
    for each mini-batch:
        model.train()
        prediction = model(x)          # forward pass
        loss = loss_fn(prediction, y)
        loss.backward()                # compute gradients
        optimizer.step()               # update weights
        optimizer.zero_grad()          # reset gradients
    validation with torch.no_grad()
```

Keras hides this inside `fit()`. In PyTorch you write it, which makes every step explicit.

## Run it

```bash
pip install torch numpy scikit-learn matplotlib tensorboard
python training_loop.py
python train_with_validation.py     # creates model_checkpoint.pth
python resume_training.py           # needs the checkpoint from the previous script
python train_with_class.py
python moons_classification.py
tensorboard --logdir logs           # or --logdir runs for the StepByStep scripts
```

All scripts run on CPU, and use a GPU automatically if one is available.

## What I learned

- The 5 steps of every PyTorch training step, and why `zero_grad()` is needed.
- `Dataset`, `DataLoader` and `random_split` for feeding data in mini-batches.
- `model.train()` vs `model.eval()`, and turning off gradients during validation.
- Saving and resuming training with checkpoints, and monitoring it in TensorBoard.
- Refactoring repeated code into a reusable class.

## Note

The `StepByStep` class follows the implementation in *Deep Learning with PyTorch Step-by-Step* by Daniel Voigt Godoy.
