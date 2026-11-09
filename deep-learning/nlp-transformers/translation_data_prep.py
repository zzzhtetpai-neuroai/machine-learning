# import os
# import urllib.request
# import zipfile

# # URL of the dataset
# url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
# zip_path = "spa-eng.zip"

# # 1. Download the file
# if not os.path.exists(zip_path):
#     print("Downloading dataset...")
#     urllib.request.urlretrieve(url, zip_path)
#     print("Download complete!")

# # 2. Extract the zip file
# extract_path = "spa-eng"
# if not os.path.exists(extract_path):
#     print("Extracting dataset...")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extract_path)
#     print("Extraction complete!")

# # 3. Check contents
# file_path = os.path.join(extract_path, "spa-eng", "spa.txt")
# print("Dataset ready at:", file_path)

# # 4. Preview a few lines
# with open(file_path, encoding="utf-8") as f:
#     for i in range(5):
#         print(f.readline().strip())



text_file='spa_eng/spa.txt'
with open(text_file,encoding='utf-8') as f:
    lines=f.read().split('\n')[:-1]
text_pairs=[]
for line in lines:
    english,spanish=line.split('\t')
    spanish='[start] '+spanish+' [end]'
    text_pairs.append((english,spanish))
import random
print(random.choice(text_pairs))

import random
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

import tensorflow as tf
import string
import re

strip_chars=string.punctuation+'¿'
strip_chars=strip_chars.replace('[','')
strip_chars=strip_chars.replace(']','')

def custom_standardization(input_string):
    lowercase=tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase,f"[{re.escape(strip_chars)}]","")

vocab_size=15000
sequence_length=20

import keras
source_vectorization=keras.layers.TextVectorization(max_tokens=vocab_size,
                                                    output_mode="int",
                                                    output_sequence_length=sequence_length)

target_vectorization=keras.layers.TextVectorization(max_tokens=vocab_size,
                                                    standardize=custom_standardization,
                                                    output_mode="int",
                                                    output_sequence_length=sequence_length+1)
train_eng_texts=[pair[0] for pair in train_pairs]
train_spa_texts=[pair[1] for pair in train_pairs]
source_vectorization.adapt((train_eng_texts))
target_vectorization.adapt((train_spa_texts))

batch_size=64

def format_dataset(eng,spa):
    eng=source_vectorization(eng)
    spa=target_vectorization(spa)
    return ({'english':eng,
             'spanish':spa[:,:-1]},
            spa[:,1:])

def make_dataset(pairs):
    eng_texts,spa_texts=zip(*pairs)
    eng_texts=list(eng_texts)
    spa_texts=list(spa_texts)
    dataset=tf.data.Dataset.from_tensor_slices((eng_texts,spa_texts))
    dataset=dataset.batch(batch_size)
    dataset=dataset.map(format_dataset,num_parallel_calls=4)
    return dataset.cache().shuffle(2048).prefetch(16)

train_dataset=make_dataset(train_pairs)
val_dataset=make_dataset(val_pairs)
