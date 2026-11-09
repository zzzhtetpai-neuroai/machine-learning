import keras.layers
import keras.models
import keras

inputs=keras.Input(shape=(28,28,1))
x=keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu')(inputs)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu')(x)
x=keras.layers.MaxPooling2D(pool_size=2)(x)
x=keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu')(x)
x=keras.layers.Flatten()(x)
outputs=keras.layers.Dense(units=10,activation='softmax')(x)
model=keras.models.Model(inputs=inputs,outputs=outputs)

from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28,28,1))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size=64)
test_loss,test_acc=model.evaluate(test_images,test_labels)  
print(f"test accuracy; {test_acc:.3f}")
