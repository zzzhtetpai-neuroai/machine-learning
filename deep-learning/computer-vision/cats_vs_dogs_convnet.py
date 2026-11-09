import keras.layers
import keras.models

inputs=keras.Input(shape=(180,180,3))
data_augmentation=keras.Sequential([keras.layers.RandomFlip('horizontal'),
                                    keras.layers.RandomRotation(0.1),
                                    keras.layers.RandomZoom(0.2),])
x=data_augmentation(inputs)
x=keras.layers.Rescaling(1./255)(x)
x=keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu')(x)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu')(x)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu')(x)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu')(x)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu')(x)
x=keras.layers.Flatten()(x)
x=keras.layers.Dropout(0.5)(x)
outputs=keras.layers.Dense(units=1,activation='sigmoid')(x)
model=keras.models.Model(inputs=inputs,outputs=outputs)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

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

for data_batch,label_batch in train_dataset:

    print(data_batch.shape)
    print(label_batch.shape)
    break

callbacks=[keras.callbacks.ModelCheckpoint(filepath='augmented_cats_and_dogs_small_model.keras',
                                           save_best_only=True,
                                           monitor='val_loss',)]

history=model.fit(train_dataset,
                  epochs=100,
                  validation_data=val_dataset,
                  callbacks=callbacks)
from matplotlib import pyplot as plt

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# model=keras.models.load_model('cats_and_dogs_small_model.keras')
# test_losss,test_accuracy=model.evaluate(test_dataset)
# print(f'Test accuracy: {test_accuracy:.3f}')
import numpy as np


plt.figure(figsize=(10,10))
for images,_ in train_dataset.take(1):
    for i in range(9):
        augmented_images=data_augmentation(images)
        ax=plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()


