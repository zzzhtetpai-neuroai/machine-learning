import keras
from keras.utils import plot_model
from keras import layers
from keras import Model

vocabulary_size=10000
tags=100
num_departments=4

input_title=keras.Input(shape=(vocabulary_size,),name="title_input")
input_text=keras.Input(shape=(vocabulary_size,),name="text_input")
input_tags=keras.Input(shape=(tags,),name="tags_input")



my_inputs=layers.Concatenate()([input_title,input_text,input_tags])
h_layer1=layers.Dense(64,activation='relu')(my_inputs)
output_priority=layers.Dense(1,activation='sigmoid',name='priority')(h_layer1)
output_department=layers.Dense(num_departments,activation='softmax',name='department')(h_layer1)
my_outputs=[output_priority,output_department]

model=Model(inputs=[input_title,input_text,input_tags],outputs=[output_priority,output_department])

import numpy as np
num_samples=1280

title_data=np.random.randint(0,2,size=((num_samples,vocabulary_size,)))
text_data=np.random.randint(0,2,size=((num_samples,vocabulary_size,)))
tags_data=np.random.randint(0,2,size=((num_samples,tags,)))
input_data=[title_data,text_data,tags_data]

priority_labels=np.random.random(size=(num_samples,1))
department_labels=np.random.randint(0,2,size=(num_samples,num_departments))
output_labels=[priority_labels,department_labels]

model.compile(optimizer='rmsprop',
              loss=['mse','categorical_crossentropy'],
              metrics=[['mean_absolute_error'],['accuracy']])
model.fit(input_data,output_labels,epochs=1)
model.evaluate(input_data,output_labels)
priority_preds,department_preds=model.predict(input_data)

# model.summary()

plot_model(model,to_file='functional_api_model.png',show_shapes=True)

# print(model.layers)
print(model.layers[3].input)
print(model.layers[3].output)

import keras
features=model.layers[4].output
difficulty_output=layers.Dense(3,activation='softmax',name='difficulty')(features)
new_model=keras.Model(inputs=[input_title,input_text,input_tags],outputs=[output_priority,output_department,difficulty_output])


keras.utils.plot_model(new_model,to_file='functional_api_model_v2.png',show_shapes=True)
