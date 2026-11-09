# Computer Vision

Convolutional neural networks (CNNs), from digit recognition to classifying cats vs dogs with a pretrained model, and pixel-level image segmentation.

## Scripts

| File | What it does |
|---|---|
| `mnist_convnet.py` | A small CNN (3 Conv2D layers with max pooling) on MNIST. Compared with the Dense network in `keras-fundamentals`, it shows why convolutions suit images |
| `unzip_cats_dogs.py` | Extracts the Kaggle Dogs vs Cats `train.zip` |
| `make_cats_dogs_subset.py` | Builds a small dataset: 1,000 train, 500 validation and 500 test images per class |
| `cats_vs_dogs_convnet.py` | A CNN trained from scratch on the small dataset, with **data augmentation** (random flip, rotation and zoom) and dropout to fight overfitting |
| `vgg16_feature_extraction.py` | **Transfer learning:** runs images through a pretrained VGG16 (ImageNet) once, then trains a small classifier on the extracted features |
| `vgg16_feature_extraction_augmented.py` | Same idea, but VGG16 is frozen inside the model so data augmentation can be used during training |
| `download_oxford_pets.py` | Downloads and extracts the Oxford-IIIT Pets images and masks |
| `image_segmentation.py` | **Image segmentation:** an encoder-decoder network (Conv2D down, Conv2DTranspose up) that labels every pixel as pet, background or outline |

## Data

The datasets are large, so they are not in this repo:

- **Dogs vs Cats:** download `train.zip` from the [Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats/data) into `data/`, then run `unzip_cats_dogs.py` and `make_cats_dogs_subset.py`.
- **Oxford-IIIT Pets:** run `download_oxford_pets.py` (about 800 MB).

## Run it

```bash
pip install tensorflow matplotlib numpy
python mnist_convnet.py
python cats_vs_dogs_convnet.py
python vgg16_feature_extraction.py
python image_segmentation.py
```

A GPU is strongly recommended for the cats vs dogs and segmentation scripts.

## What I learned

- Why convolutions and pooling work for images: local patterns, and translation invariance.
- How data augmentation and dropout reduce overfitting on small datasets.
- Transfer learning: reusing features learned on ImageNet instead of starting from zero.
- The difference between classifying a whole image and segmenting every pixel.
