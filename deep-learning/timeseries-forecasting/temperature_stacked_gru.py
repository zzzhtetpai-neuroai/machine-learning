import os

filename=os.path.join("RNN","jena_climate_2009_2016.csv")
with open(filename,'r') as f:
    data=f.read()
    lines=data.split("\n")
    header=lines[0].split(",") #list.split is not avalilable u can only split string
    lines=lines[1:]
    print(header)
    print(len(lines))

import numpy as np
raw_data=np.zeros((len(lines),len(header)-1)) #slice off time and date header 
temperature=np.zeros((len(lines),))
for i,line in enumerate(lines):
    value=[float(x) for x in line.split(",")[1:] ] #slice off the time and date column
    temperature[i]=value[1]
    raw_data[i,:]=value[:]

# import matplotlib.pyplot as plt
# plt.plot(range(1440),temperature[:1440])
# plt.show()
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)


mean=raw_data[:num_train_samples].mean(axis=0)
raw_data-=mean
std=raw_data[:num_train_samples].std(axis=0)
raw_data/=std

#separating data
import keras.utils
sampling_rate=6
sequence_length=120
delay=sampling_rate*(sequence_length+24-1)
batch_size=256

train_dataset=keras.utils.timeseries_dataset_from_array(raw_data[:-delay],
                                                        targets=temperature[delay:],
                                                        sampling_rate=sampling_rate,
                                                        sequence_length=sequence_length,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        start_index=0,
                                                        end_index=num_train_samples)

val_dataset=keras.utils.timeseries_dataset_from_array(raw_data[:-delay],
                                                        targets=temperature[delay:],
                                                        sampling_rate=sampling_rate,
                                                        sequence_length=sequence_length,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        start_index=num_train_samples,
                                                        end_index=num_train_samples+num_val_samples)

test_dataset=keras.utils.timeseries_dataset_from_array(raw_data[:-delay],
                                                        targets=temperature[delay:],
                                                        sampling_rate=sampling_rate,
                                                        sequence_length=sequence_length,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        start_index=num_train_samples+num_val_samples)



#RNN layers
import keras
from keras import models


inputs=keras.Input(shape=(sequence_length,raw_data.shape[-1]))
x=keras.layers.GRU(32,recurrent_dropout=0.5,return_sequences=True)(inputs)
x=keras.layers.GRU(32,recurrent_dropout=0.5)(x)
x=keras.layers.Dropout(0.5)(x)
outputs=keras.layers.Dense(1)(x)
model=models.Model(inputs,outputs)

callbacks=[keras.callbacks.ModelCheckpoint(filepath='RNN_dropout.keras',
                                           save_best_only=True)]

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
history=model.fit(train_dataset,
                  epochs=50,
                  validation_data=val_dataset,
                  callbacks=callbacks)


model=keras.models.load_model('RNN_dropout.keras')
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")