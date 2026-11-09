import numpy
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetworks:
    def __init__(self,inodes,hnodes,onodes,learning_rate):
        self.inodes=inodes
        self.hnodes=hnodes
        self.onodes=onodes
        self.w_ih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.w_ho=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learning_rate
        self.activation_function=lambda x:scipy.special.expit(x)
    
    def train(self,input_lists,target_lists):
        inputs=numpy.array(input_lists,ndmin=2).T
        targets=numpy.array(target_lists,ndmin=2).T
        hidden_inputs=numpy.dot(self.w_ih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.w_ho,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.w_ho.T,output_errors)
        self.w_ho+=self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))
        self.w_ih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),numpy.transpose(inputs))
        pass
    


    def query(self,input_lists):
        inputs=numpy.array(input_lists,ndmin=2).T
        hidden_inputs=numpy.dot(self.w_ih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.w_ho,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs

input_nodes=784
hidden_nodes=200
output_nodes=10
learning_rate=0.1
n=NeuralNetworks(input_nodes,hidden_nodes,output_nodes,learning_rate)


training_data_file=open("dataset/mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()
epochs=5
for e in range(epochs):
    for record in training_data_list:
        all_values=record.split(",")
        inputs=numpy.asarray(all_values[1:],dtype='float32')/255*0.99+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass


test_data_file=open("dataset/mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()
all_values=test_data_list[0].split(",")
# print(all_values[0])
# image_array=numpy.asarray(all_values[1:],dtype='float32').reshape((28,28))
# plt.imshow(image_array,cmap='Greys',interpolation='None')
# plt.show()
raw_pixels = numpy.asarray(all_values[1:], dtype='float32').reshape((28*28 ))
inputs = raw_pixels/ 255.0 * 0.99 + 0.01
outputs=n.query(inputs)
# print(outputs)
scorecard=[]

for record in test_data_list:
    all_values=record.split(",")
    correct_label=int(all_values[0])
    # print(correct_label,"correct label")
    inputs=numpy.asarray(all_values[1:],dtype='float32')/255*0.99+0.01
    outputs=n.query(inputs)
    label=numpy.argmax(outputs)
    # print(label,"network's answer")
    if(label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
# print(scorecard)
pass

scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)