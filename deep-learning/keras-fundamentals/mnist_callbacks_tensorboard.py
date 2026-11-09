from keras.datasets import mnist
from keras.layers import Dense,Input
from keras.models import Model
from keras.optimizers import Adam
import keras

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

def build_model():
    inputs=Input(shape=(28*28,))
    hidden1=Dense(512,activation='relu')(inputs)
    hidden2=Dense(256,activation='relu')(hidden1)
    hidden3=Dense(128,activation='relu')(hidden2)
    outputs=Dense(10,activation='softmax')(hidden3)
    model=Model(inputs=inputs,outputs=outputs)
    return model

val_data=train_images[:10000]
partial_train_data=train_images[10000:]
val_labels=train_labels[:10000]
partial_train_labels=train_labels[10000:]
es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
mc=keras.callbacks.ModelCheckpoint('best_model.keras',monitor='val_accuracy',save_best_only=True)
tensorboard=keras.callbacks.TensorBoard(log_dir='./logs')
callbacks=[es,mc,tensorboard]


model=build_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_train_data,partial_train_labels,
          epochs=20,
          batch_size=512,
          validation_data=(val_data,val_labels),
            callbacks=callbacks
            )

