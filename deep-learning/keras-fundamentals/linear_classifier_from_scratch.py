import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples_per_class=1000
negative_samples=np.random.multivariate_normal(
    mean=[0,3],
    cov=[[1,0.5],[0.5,1]],
    size=num_samples_per_class
)
positive_samples=np.random.multivariate_normal(
    mean=[3,0],
    cov=[[1,0.5],[0.5,1]],
    size=num_samples_per_class
)
inputs=np.vstack((negative_samples,positive_samples)).astype(np.float32)

targets=np.vstack((np.zeros((num_samples_per_class,1)),np.ones((num_samples_per_class,1)))).astype(np.float32)

input_dim=2
output_dim=1
W=tf.Variable(tf.random.normal(shape=(input_dim,output_dim)))
b=tf.Variable(tf.zeros(shape=(output_dim,)))

def model(inputs):
    return tf.matmul(inputs,W)+b

def square_loss(targets,predictions):
    return tf.reduce_mean(tf.square(targets-predictions))

learning_rate=0.1

def training_step(inputs,targets):
    with tf.GradientTape() as tape:
        predictions=model(inputs)
        loss=square_loss(targets,predictions)
    grad_W,grad_b=tape.gradient(loss,[W,b])
    W.assign_sub(learning_rate*grad_W)
    b.assign_sub(learning_rate*grad_b)
    return loss

for step in range(40):
    loss=training_step(inputs,targets)
    print(f'Step {step+1}, Loss: {loss :.4f}')

predictions=model(inputs)
plt.scatter(inputs[:,0],inputs[:,1],c=predictions[:,0]>0.5)
plt.show()

x=np.linspace(-1,4,100)
y=-(W[0]/W[1]) * x + ((0.5-b)/W[1])
plt.plot(x,y,color='red')
plt.scatter(inputs[:,0],inputs[:,1],c=predictions[:,0]>0.5)
plt.show()