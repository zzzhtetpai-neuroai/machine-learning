#get our data
from imdb_binary_classification import vectorize
from keras.datasets import reuters
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

#process our data
import numpy as np
import keras
x_train=vectorize(train_data)
x_test=vectorize(test_data)

# y_test=keras.utils.to_categorical(test_labels) # one hot encoding which uses categorical_crossentropy
# y_train=keras.utils.to_categorical(train_labels) # one hot encoding which uses categorical_crossentropy

y_train=np.asarray(train_labels) #integer encoding which uses sparse_categorical_crossentropy
y_test=np.asarray(test_labels) #integer encoding which uses sparse_categorical_crossentropy

#build our model
model=keras.Sequential([
    keras.layers.Dense(units=64,activation='relu'),
    keras.layers.Dense(units=64,activation='relu'),
    keras.layers.Dense(units=46,activation='softmax'),
    ])
#compile our model
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )
#validate our model
x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=y_train[:1000]
partial_y_train=y_train[1000:]
#train our model
model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))

#visualize our data
import matplotlib.pyplot as plt
history_dict=model.history.history
loss_values=history_dict['loss']        
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#evaluate our model
results=model.evaluate(x_test,y_test)
print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")