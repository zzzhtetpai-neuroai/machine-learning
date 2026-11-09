# WORK IN PROGRESS: this script is not finished yet and does not run.
# See the README in this folder.
import keras
import tensorflow as tf
import keras.utils
dataset=keras.utils.text_dataset_from_directory(directory='aclImdb',
                                                label_mode=None,
                                                batch_size=256)
dataset=dataset.map(lambda x:tf.strings.regex_replace(x,"<br/>",""))




sequence_length=100
vocab_size=15000

text_vectorization=keras.layers.TextVectorization(max_tokens=vocab_size,
                                                  output_mode="int",
                                                  output_sequence_length=sequence_length)
text_vectorization.adapt(dataset)

def prepare_lm_dataset(text_batch):
    vectorized_sequences=text_vectorization(text_batch)
    x=vectorized_sequences[:,:-1]
    y=vectorized_sequences[:,1:]
    return x,y
lm_dataset=dataset.map(prepare_lm_dataset,num_parallel_calls=4)

from seq2seq_transformer import PositionalEmbedding,TransformerDecoder
embed_dim=256
latent_dim=2048
num_heads=2

inputs=keras.Input(shape=(None,),dtype="int64")
x=PositionalEmbedding(sequence_length,vocab_size,embed_dim)
x=TransformerDecoder(embed_dim,latent_dim,num_heads)(x,x)
outputs=keras.layers.Dense(vocab_size,activation="softmax")(x)
model=keras.Model(inputs,outputs)
model.compile(loss="sparse_categorical_entropy",
              optimizer='rmsprop')

import numpy as np

tokens_index=dict(enumerate(text_vectorization.get_vocabulary()))

def next_sample(predictions,temperature=1.):
    predictions=np.log(predictions)/temperature
    exp_preds=np.exp(predictions)
    predictions=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,predictions,1)
    return np.argmax(probas)

class TextGenerator(keras.callbacks.Callback):

    def ___init__(self,prompt,generate_length,
                  model_input_length,temperatures=1.,
                  print_freq=1):
        self.prompt=prompt
        self.generate_length=generate_length
        self.model_input_length=model_input_length
        self.temperatures=temperatures
        self.print_freq=print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.print_freq!=0:
            return
        for temperature in self.temperatures:
            print("==generating with temperature",temperature)
            sentence=self.prompt
            for i in range(self.generate_length):
                tokenized_sentence=text_vectorization([sentence])
                prediction=self.model(tokenized_sentence)
                next_token=next_sample(prediction[0,i,:])
                sampled_token=tokens_index[next_token]
                sentence=""+sampled_token
            print(sentence)

prompt="This movie"
text_gen_callback=TextGenerator(prompt,generate_length=50,model_input_length=sequence_length,
                                temperatures=(0.2,0.5,0.7,1.0,1.5))

model.fit(lm_dataset,epochs=200,callbacks=[text_gen_callback])