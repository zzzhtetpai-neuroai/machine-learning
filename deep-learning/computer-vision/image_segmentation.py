import os

input_dir = "images/"
target_dir = "annotations/trimaps/"

# Collect all image paths ending with .jpg
input_img_paths = sorted([
    os.path.join(input_dir, fname)
    for fname in os.listdir(input_dir)
    if fname.endswith(".jpg")
])

# Collect all mask paths ending with .png (ignore hidden files)
target_paths = sorted([
    os.path.join(target_dir, fname)
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
])
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array

plt.axis("off")  # hides the x/y axis ticks and labels
plt.imshow(load_img(input_img_paths[9]))  # loads and shows the 10th image
plt.show()

def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array

# --- Config ---
input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (200, 200)
val_count = 1000
seed = 1337

# --- Collect paired paths (to avoid misalignment) ---
input_img_paths = sorted([
    os.path.join(input_dir, fname)
    for fname in os.listdir(input_dir)
    if fname.endswith(".jpg")
])

target_paths = sorted([
    os.path.join(target_dir, fname)
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
])

# Sanity check
assert len(input_img_paths) == len(target_paths), "Image/mask count mismatch."

# Pair and shuffle together
pairs = list(zip(input_img_paths, target_paths))
random.Random(seed).shuffle(pairs)
input_img_paths, target_paths = zip(*pairs)  # unzip back to two lists

num_imgs = len(input_img_paths)

# --- Loaders ---
def path_to_input_image(path):
    # RGB image -> float32 array (H, W, 3)
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    # Grayscale mask -> uint8 array (H, W, 1), labels {1,2,3} -> {0,1,2}
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img

# --- Preallocate arrays ---
input_imgs = np.zeros((num_imgs, img_size[0], img_size[1], 3), dtype="float32")
targets = np.zeros((num_imgs, img_size[0], img_size[1], 1), dtype="uint8")

# --- Load all into memory ---
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

# --- Train/val split ---
assert val_count < num_imgs, "Validation count must be less than total images."
train_input_imgs = input_imgs[:-val_count]
train_targets = targets[:-val_count]

val_input_imgs = input_imgs[-val_count:]
val_targets = targets[-val_count:]

# --- Quick visualization (one sample) ---
idx = 9  # 10th sample after shuffle
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Input image")
plt.axis("off")
plt.imshow(input_imgs[idx].astype("uint8"))

plt.subplot(1, 2, 2)
plt.title("Target mask")
plt.axis("off")
# Scale {0,1,2} -> {0,127,254} for visualization
mask_vis = (targets[idx].astype("uint8") * 127)[:, :, 0]
plt.imshow(mask_vis, cmap="gray")
plt.tight_layout()
plt.show()

# print(f"Total: {num_imgs} | Train: {len(train_input_imgs)} | Val: {len(val_input_imgs)}")
# print(f"Input shape: {input_imgs.shape} | Target shape: {targets.shape}")
import keras
from keras import layers

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)

    # Encoder (downsampling)
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    # Decoder (upsampling)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)

    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)

    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

    # Output layer (softmax for segmentation classes)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

# Example usage
img_size = (200, 200)
model = get_model(img_size=img_size, num_classes=3)
model.summary()
# --- Compile the model ---
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy"
)

# --- Callbacks ---
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "oxford_segmentation.keras",
        save_best_only=True
    )
]

# --- Train the model ---
history = model.fit(
    train_input_imgs,
    train_targets,
    epochs=50,
    callbacks=callbacks,
    batch_size=64,
    validation_data=(val_input_imgs, val_targets)
)

# --- Plot training and validation loss ---
import matplotlib.pyplot as plt

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")       # blue dots
plt.plot(epochs, val_loss, "b", label="Validation loss")  # blue line
plt.title("Training and validation loss")
plt.legend()
plt.show()
