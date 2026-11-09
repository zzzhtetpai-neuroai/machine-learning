# ResNet Image Classification

Classifying an image with **ResNet-18**, a convolutional neural network pretrained on ImageNet (1,000 classes), using PyTorch.

## The idea

Training an image model from scratch needs millions of images and lots of GPU time. A pretrained model has already learned general visual features, so you can load it and classify new images right away. This is the starting point for transfer learning.

## How it works

1. Load `resnet18` with its default ImageNet weights and switch it to evaluation mode.
2. Preprocess the image the same way as during ImageNet training:
   resize to 256 → center crop 224 → convert to tensor → normalize with the ImageNet mean and std.
3. Run inference with `torch.no_grad()` (no gradients needed).
4. Take the class with the highest score and map it to its name using the official ImageNet label list.

## Example

The script classifies `dog.jpg` and prints the predicted ImageNet class index and label (for example, a dog breed).

To try your own image, put it in this folder and change the file name in `Image.open("dog.jpg")`.

## Run it

```bash
pip install torch torchvision pillow requests
python pretrained_model.py
```

The first run downloads the ResNet-18 weights (about 45 MB) and the ImageNet label list.

## What I learned

- Using pretrained models from `torchvision` instead of training from scratch.
- Why input preprocessing has to match what the model was trained on.
- The difference between training mode and evaluation mode (`model.eval()`).
