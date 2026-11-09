#first lets get our data
from keras.datasets import imdb


(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

# word_index = imdb.get_word_index()                                
# reverse_word_index = dict(
#  [(value, key) for (key, value) in word_index.items()])        
# decoded_review = " ".join(
#  [reverse_word_index.get(i - 3, "?") for i in train_data[0]]) 


#lets process our data
import numpy as np

def vectorize (sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        for j in sequence:
            results[i][j]=1.
    return results


if __name__=="__main__":
    x_train=vectorize(train_data)
    x_test=vectorize(test_data)

    y_train=np.asarray(train_labels).astype('float32')
    y_test=np.asarray(test_labels).astype('float32')


#lets build our model
#model,compile,fit(add validate),evaluate
    import keras
    from keras.layers import Dense
    from keras.optimizers import Adam


    model=keras.Sequential([
                            Dense(units=16,activation='relu'),
                            Dense(units=16,activation='relu'),
                            Dense(units=1,activation='sigmoid'),
                        ])
    model.compile(
                optimizer=Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy'],
                )
#lets validate our modle
    x_val=x_train[:10000]
    partial_x_train=x_train[10000:]
    y_val=y_train[:10000]   
    partial_y_train=y_train[10000:]
#lets train our model
    model.fit(partial_x_train,partial_y_train,epochs=4,batch_size=512,validation_data=(x_val,y_val))


#lets visualize our data
    import matplotlib.pyplot as plt

    history_dict=model.history.history
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    x_ticks=range(1,len(loss_values)+1)
    plt.plot(x_ticks,loss_values,'bo',label='training loss')
    plt.plot(x_ticks,val_loss_values,'b',label='validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    plt.clf()
    acc_values=history_dict['accuracy']
    val_acc_values=history_dict['val_accuracy']
    plt.plot(x_ticks,acc_values,'bo',label='training accuracy')
    plt.plot(x_ticks,val_acc_values,'b',label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#lets evaluate our model
    results=model.evaluate(x_test,y_test)
    print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")  