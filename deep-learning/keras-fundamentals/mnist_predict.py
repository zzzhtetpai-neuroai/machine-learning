from keras.datasets import mnist
from keras.models import load_model

_,(test_images,test_labels)=mnist.load_data()
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
model=load_model('mnist_model.keras')
test_digits=test_images[:10]
predicitions=model.predict(test_digits)
print(predicitions[0])
print(predicitions[0].argmax())
print(test_labels[0])


test_loss,test_accuracy=model.evaluate(test_images,test_labels)
print(f"Test accuracy is {test_accuracy}")
