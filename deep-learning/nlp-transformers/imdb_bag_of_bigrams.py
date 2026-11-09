import os,pathlib,shutil,random

base_dir=pathlib.Path("aclImdb")
val_dir=base_dir/"val"
train_dir=base_dir/"train"

# for category in ("neg","pos"):
#     os.makedirs(val_dir/category)
#     files=os.listdir(train_dir/category)
#     random.Random(1337).shuffle(files)
#     num_val=int(0.2*len(files))
#     val_files=files[-num_val:]
#     for fname in val_files:
#         shutil.move(train_dir/category/fname,val_dir/category/fname)

import keras
batch_size=32

train_dataset=keras.utils.text_dataset_from_directory(train_dir,batch_size=batch_size)
val_dataset=keras.utils.text_dataset_from_directory(val_dir,batch_size=batch_size)
test_dataset=keras.utils.text_dataset_from_directory(base_dir/"test",batch_size=batch_size)


text_vectorization=keras.layers.TextVectorization(ngrams=2,max_tokens=20000,output_mode='tf_idf')
text_only_train_dataset=train_dataset.map(lambda x,y:x)
text_vectorization.adapt(text_only_train_dataset)
binary_1_gram_train_dataset=train_dataset.map(lambda x,y:(text_vectorization(x),y),num_parallel_calls=4)
binary_1_gram_val_dataset=val_dataset.map(lambda x,y:(text_vectorization(x),y),num_parallel_calls=4)
binary_1_gram_test_dataset=test_dataset.map(lambda x,y:(text_vectorization(x),y),num_parallel_calls=4)



def get_model(max_tokens=20000,hidden_dim=16):
    inputs=keras.Input(shape=(max_tokens,))
    x=keras.layers.Dense(hidden_dim,activation='relu',)(inputs)
    x=keras.layers.Dropout(0.5)(x)
    outputs=keras.layers.Dense(1,activation='sigmoid')(x)
    model=keras.Model(inputs,outputs)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model=get_model()
model.summary()
callbacks=keras.callbacks.ModelCheckpoint(save_best_only=True,
                                          filepath='zNLP.keras',)
model.fit(binary_1_gram_train_dataset.cache(),
          validation_data=binary_1_gram_val_dataset.cache(),
          epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("zNLP.keras") 
print(f"Test acc: {model.evaluate(binary_1_gram_test_dataset)[1]:.3f}")


#when we want to export put text vectorization inside the model like this

#inputs=keras.Input(shape=(1,),dtype="string")
#inputs=text_vectorization(inputs) #we have already defined this object