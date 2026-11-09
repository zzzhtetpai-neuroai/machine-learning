from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#process our data (here we need to scale our data bc features have different ranges)

mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std
test_data-=mean  #never find mean using test data bc it is unseen data(cheating)
test_data/=std

#build our model
from keras import Sequential
from keras.layers import Dense
def build_model():
    model=Sequential([
                    Dense(64,activation='relu'),
                    Dense(64,activation='relu'),
                    Dense(1), #no activation function for regression
        
                    ])
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'],
                  )
    return model

#validate our model using k-fold validation (we have small dataset)
import numpy as np
k=4
num_val_samples=len(train_data)//k
num_epochs=100
all_mae_histories=[]
for i in range(k):
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],
                                      train_data[(i+1)*num_val_samples:]],
                                      axis=0
                                      )   
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],
                                         train_targets[(i+1)*num_val_samples:]],
                                         axis=0,
                                         )
    model=build_model()
    history=model.fit(partial_train_data,
                      partial_train_targets,
                      validation_data=(val_data,val_targets),
                      epochs=num_epochs,
                      batch_size=16,
                      verbose=0,
                      )
    val_mae=history.history['val_mae']
    all_mae_histories.append(val_mae)

    #visualize our validation scores
average_mae_history=[np.mean([x[i]for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1,num_epochs+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
    


model=build_model()
model.fit(train_data,train_targets,epochs=130,batch_size=16,verbose=0)
test_mse,test_mae=model.evaluate(test_data,test_targets)
print(f"Test MAE: {test_mae:.2f} (thousand dollars)")
