import os,pathlib,shutil,random

base_dir=pathlib.Path("aclImdb")
val_dir=base_dir/"val"
train_dir=base_dir/"train"
import keras
batch_size=32

train_dataset=keras.utils.text_dataset_from_directory(train_dir,batch_size=batch_size)
val_dataset=keras.utils.text_dataset_from_directory(val_dir,batch_size=batch_size)
test_dataset=keras.utils.text_dataset_from_directory(base_dir/"test",batch_size=batch_size)


text_only_train_dataset=train_dataset.map(lambda x,y:x)


from keras import layers
max_length = 600 
max_tokens = 20000 
text_vectorization = layers.TextVectorization(max_tokens=max_tokens,
                                              output_mode="int",
                                              output_sequence_length=max_length,     
                                              )
text_vectorization.adapt(text_only_train_dataset)
int_train_ds = train_dataset.map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)
int_val_ds = val_dataset.map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)
int_test_ds = test_dataset.map(lambda x, y: (text_vectorization(x), y),num_parallel_calls=4)

import tensorflow as tf
inputs = keras.Input(shape=(None,), dtype="int64")    
embedded = tf.one_hot(inputs, depth=max_tokens)       
x = layers.Bidirectional(layers.LSTM(32))(embedded)   
x = layers.Dropout(0.5)(x) 
outputs = layers.Dense(1, activation="sigmoid")(x)    
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
                                    save_best_only=True)
                                    ]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10,callbacks=callbacks)
model = keras.models.load_model("one_hot_bidir_lstm.keras") 
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")