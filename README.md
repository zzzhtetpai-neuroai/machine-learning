# Machine Learning

My hands-on path through machine learning, from simple bandit algorithms to deep Q-networks that learn to drive, play Snake and save energy.

Every folder is a self-contained project with its own README that explains the idea, how the code works, how to run it and what I learned.

## Projects

### Reinforcement learning

| Project | Technique | Framework |
|---|---|---|
| [Thompson Sampling](reinforcement-learning/thompson-sampling) | Multi-armed bandit, compared with random selection | NumPy, Matplotlib |
| [Q-Learning Warehouse Routing](reinforcement-learning/q-learning) | Tabular Q-learning for shortest routes | NumPy |
| [Self-Driving Car](reinforcement-learning/self-driving-car) | Deep Q-Network with experience replay | PyTorch, Kivy |
| [Snake DQN](reinforcement-learning/snake-dqn) | CNN-based Deep Q-Network on game frames | Keras, Pygame |
| [Energy Optimization DQN](reinforcement-learning/energy-optimization-dqn) | Deep Q-Network for server cooling control | Keras |

### Deep learning

| Project | Technique | Framework |
|---|---|---|
| [House Price Regression](deep-learning/house-price-regression) | Feedforward neural network for regression | Keras, scikit-learn |
| [ResNet Image Classification](deep-learning/resnet-image-classification) | Inference with a pretrained ResNet-18 | PyTorch, torchvision |

## Repository structure

```
machine-learning/
├── reinforcement-learning/
│   ├── thompson-sampling/
│   ├── q-learning/
│   ├── self-driving-car/
│   ├── snake-dqn/
│   └── energy-optimization-dqn/
└── deep-learning/
    ├── house-price-regression/
    └── resnet-image-classification/
```

## Running a project

Each project has its own dependencies, listed in its README. The scripts load and save files with relative paths, so run them from inside the project folder:

```bash
git clone https://github.com/zzzhtetpai-neuroai/machine-learning.git
cd machine-learning/reinforcement-learning/q-learning
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install numpy
python q_learning.py
```

## Learning resources

Most projects here were built while working through **AI Crash Course** by Hadelin de Ponteves (Packt, 2019). I typed and ran the code myself, studied how each part works and changed it along the way. Each README notes where a project comes from and what I changed.

## Author

Htet Pai, IT Business student at iU (Tokyo), on a path from cloud and ML toward brain-computer interfaces.
GitHub: [@zzzhtetpai-neuroai](https://github.com/zzzhtetpai-neuroai)
