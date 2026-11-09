# Machine Learning

My hands-on machine learning projects, from classic algorithms to deep learning and reinforcement learning, built while learning each topic step by step.

Every project folder has its own README that explains the idea, how the code works, how to run it and what I learned.

## Projects

### Reinforcement Learning

| Project | Technique | Framework |
|---|---|---|
| [Thompson Sampling](reinforcement-learning/thompson-sampling) | Multi-armed bandit, compared with random selection | NumPy, Matplotlib |
| [Q-Learning Warehouse Routing](reinforcement-learning/q-learning) | Tabular Q-learning for shortest routes | NumPy |
| [Self-Driving Car](https://github.com/zzzhtetpai-neuroai/self-driving-car-reinforcement-learning) ↗ | Deep Q-Network with experience replay | PyTorch, Kivy |
| [Snake DQN](https://github.com/zzzhtetpai-neuroai/snake-game-cnn-deep-q-learning) ↗ | CNN-based Deep Q-Network on game frames | Keras, Pygame |
| [Energy Optimization DQN](https://github.com/zzzhtetpai-neuroai/energy-optimization-deep-q-learning) ↗ | Deep Q-Network for server cooling control | Keras |

### Deep Learning

| Project | Technique | Framework |
|---|---|---|
| [House Price Regression](deep-learning/house-price-regression) | Feedforward neural network for regression | Keras, scikit-learn |
| [ResNet Image Classification](deep-learning/resnet-image-classification) | Inference with a pretrained ResNet-18 | PyTorch, torchvision |
| [Keras Fundamentals](deep-learning/keras-fundamentals) | First networks, training from scratch, Sequential and Functional APIs, callbacks | TensorFlow, Keras |
| [Classification and Regression](deep-learning/classification-and-regression) | Binary, multiclass and regression on IMDB, Reuters and Boston Housing | Keras |
| [Computer Vision](deep-learning/computer-vision) | CNNs, data augmentation, transfer learning with VGG16, image segmentation | Keras |
| [Time Series Forecasting](deep-learning/timeseries-forecasting) | LSTM, stacked GRU and bidirectional RNNs for temperature prediction | Keras |
| [NLP and Transformers](deep-learning/nlp-transformers) | Bag of words, LSTMs, Transformer encoder, encoder-decoder translation | Keras, TensorFlow |
| [Generative Deep Learning](deep-learning/generative) | DeepDream with InceptionV3 | Keras |
| [Neural Network from Scratch](https://github.com/zzzhtetpai-neuroai/neural-network-from-scratch) ↗ | MNIST digit classifier with hand-written backpropagation | NumPy |
| [PyTorch Fundamentals](deep-learning/pytorch-fundamentals) | Hand-written training loops, checkpoints, TensorBoard, a reusable trainer class | PyTorch |

↗ = complete project with its own repository.

*More projects are being added.*

## Running a project

Each project lists its own dependencies in its README. The scripts load and save files with relative paths, so run them from inside the project folder:

```bash
git clone https://github.com/zzzhtetpai-neuroai/machine-learning.git
cd machine-learning/reinforcement-learning/q-learning
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install numpy
python q_learning.py
```

## Learning resources

These projects were built while studying the following books. I typed and ran the code, reviewed it, fixed bugs and reorganized it.

- *AI Crash Course* by Hadelin de Ponteves (Packt)
- *Deep Learning with Python* by François Chollet (Manning)
- *Make Your Own Neural Network* by Tariq Rashid
- *Deep Learning with PyTorch Step-by-Step* by Daniel Voigt Godoy

## Author

Htet Pai, IT Business student at iU (Tokyo)
GitHub: [@zzzhtetpai-neuroai](https://github.com/zzzhtetpai-neuroai)
