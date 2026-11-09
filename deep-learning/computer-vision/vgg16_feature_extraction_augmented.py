#feature extraction using pretrained model

import keras
from keras.utils import image_dataset_from_directory

train_dataset=image_dataset_from_directory('data/cats_and_dogs_small/train',
                                           image_size=(180,180),
                                           batch_size=32)
val_dataset=image_dataset_from_directory('data/cats_and_dogs_small/validation',
                                         image_size=(180,180),
                                         batch_size=32)
test_dataset=image_dataset_from_directory('data/cats_and_dogs_small/test',
                                          image_size=(180,180),
                                          batch_size=32)

conv_base=keras.applications.vgg16.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_shape=(180,180,3))
conv_base.trainable=False

import numpy as np

# def get_features_and_labels(dataset):
#     all_features=[]
#     all_labels=[]
#     for image,label in dataset:
#         preprocess_images=keras.applications.vgg16.preprocess_input(image)
#         features=conv_base.predict(preprocess_images)
#         all_features.append(features)
#         all_labels.append(label)
#     return np.concatenate(all_features),np.concatenate(all_labels)

# train_features,train_labels=get_features_and_labels(train_dataset)
# val_features,val_labels=get_features_and_labels(val_dataset)
# test_features,test_labels=get_features_and_labels(test_dataset)

# print(test_features.shape)

inputs=keras.Input(shape=(180,180,3))
data_augmentation=keras.Sequential([keras.layers.RandomFlip('horizontal'),
                                    keras.layers.RandomRotation(0.1),
                                    keras.layers.RandomZoom(0.2),])
x=data_augmentation(inputs)
x=keras.applications.vgg16.preprocess_input(x)
x=conv_base(x)
x=keras.layers.Flatten()(x)
x=keras.layers.Dense(256)(x)
x=keras.layers.Dropout(0.5)(x)
outputs=keras.layers.Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint(filepath="feature_extraction_with_augmentation.keras",
                                            save_best_only=True,
                                            monitor="val_loss")]
history = model.fit(
                    train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)


import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()