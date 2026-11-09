from keras import Sequential
from keras import layers
import keras

model=Sequential(name="my_model")
model.add(keras.Input(shape=(None,3),name="input_layer"))
model.add(layers.Dense(64,activation='relu',name="hidden_layer"))
model.add(layers.Dense(10,activation='softmax',name="output_layer"))
print(model.weights)
print(model.summary())