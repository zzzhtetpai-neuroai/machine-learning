import keras
import matplotlib.pyplot as plt

base_image_path=keras.utils.get_file("coast.jpg",origin="https://img-datasets.s3.amazonaws.com/coast.jpg")
plt.axis("off")
plt.imshow(keras.utils.load_img(base_image_path))
plt.show()

import tensorflow as tf
from keras.applications import inception_v3
model=inception_v3.InceptionV3(weights="imagenet",include_top=False)

layer_settings={
    "mixed4":2.0,
    "mixed5":2.5,
    "mixed6":3.0,
    "mixed7":4.5,
}

outputs_dict=dict([
    (layer.name,layer.output) for layer in [model.get_layer(name) for name in layer_settings.keys()]
])

feature_extractor=keras.Model(inputs=model.inputs,outputs=outputs_dict)

def compute_loss(input_image):
    features=feature_extractor(input_image)
    loss=tf.zeros(shape=())
    for name in features.keys():
        coeff=layer_settings[name]
        activation=features[name]
        loss+=coeff*(tf.reduce_mean(tf.square(activation[:,2:-2,2:-2,:])))
    return loss

@tf.function
def gradient_ascent_step(image,learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss=compute_loss(image)
        grads=tape.gradient(loss,image)
        grads=tf.math.l2_normalize(grads)
        image+=grads*learning_rate
        return loss,image

def gradient_ascent_loop(image,iterations,learning_rate,max_loss=None):
    for i in range(iterations):
        loss,image=gradient_ascent_step(image,learning_rate)
        if max_loss is not None and loss>max_loss:
            break
        print(f"loss at step {i}: {loss:.2f}")
    return image

import numpy as np


def preprocess_image(image_path):
    image=keras.utils.load_img(image_path)
    image=keras.utils.img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=keras.applications.inception_v3.preprocess_input(image)
    return image

def deprocess_image(image):
    image=image.reshape((image.shape[1],image.shape[2],3))
    image/=2.0
    image+=0.5
    image*=255
    image=np.clip(image,0,255).astype("uint8")
    return image

step = 20.           
num_octave = 3       
octave_scale = 1.4   
iterations = 30      
max_loss = 15.

original_image=preprocess_image(base_image_path)
original_shape=original_image.shape[1:3]

successive_shapes=[original_shape]
for i in range(num_octave):
    shape=tuple([int(dim/octave_scale**i) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes=successive_shapes[::-1]
shrunk_original_image=tf.image.resize(original_image,successive_shapes[0])
img=tf.identity(original_image)
for i,shape in enumerate(successive_shapes):
    print(f"Processing octave {i} with shape {shape}")
    img=tf.image.resize(img,shape)
    img=gradient_ascent_loop(img,iterations=iterations,learning_rate=step,max_loss=max_loss)
    upscaled_shrunk_image=tf.image.resize(shrunk_original_image,shape)
    same_size_original=tf.image.resize(original_image,shape)
    loss_detail=same_size_original-upscaled_shrunk_image
    img+=loss_detail
    shrunk_original_image=tf.image.resize(original_image,shape)

keras.utils.save_img("dream.png",deprocess_image(img.numpy()))

