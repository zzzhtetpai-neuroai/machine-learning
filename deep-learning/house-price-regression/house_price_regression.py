import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam



url="https://raw.githubusercontent.com/PacktPublishing/AI-Crash-Course/master/Chapter%2009/kc_house_data.csv"
dataset=pd.read_csv(url)
# print(dataset.head())

#Preparing the data
X=dataset.iloc[:,3:].values
X=X[:,np.r_[0:13,14:18]]
y=dataset.iloc[:,2].values

#Splitting the data for training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Feature Scaling
xscaler=MinMaxScaler(feature_range=(0,1))
X_train=xscaler.fit_transform(X_train)
X_test=xscaler.transform(X_test)

#target scaling
yscaler=MinMaxScaler(feature_range=(0,1))
y_train=yscaler.fit_transform(y_train.reshape(-1,1))#create a fake dimension to avoid errors in format
y_test=yscaler.transform(y_test.reshape(-1,1))


#Building the neural network
model=Sequential()
model.add(Dense(units=64,kernel_initializer='uniform',activation='relu',input_dim=17))
model.add(Dense(units=16,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='relu'))
model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=['mean_absolute_error'])


#Training the model
model.fit(X_train,y_train,batch_size=32,epochs=100,validation_data=(X_test,y_test))

#Making predictions
y_test=yscaler.inverse_transform(y_test)
prediction=yscaler.inverse_transform(model.predict(X_test))


error=abs(y_test-prediction)/y_test
print(np.mean(error))