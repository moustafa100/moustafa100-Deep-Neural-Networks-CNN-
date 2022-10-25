#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation,Dropout,Conv2D,MaxPool2D
from tensorflow.keras.layers import Flatten,concatenate,Input
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import mnist
import graphviz
import pydot


# In[2]:


# load datasets (mnist)
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[3]:


#compute the number of labels
nu_labels=len(np.unique(y_train))


# In[4]:


#convert into one_hot vector
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[5]:


# input image dimensions
img_size=x_train.shape[1]


# In[6]:


# resize and normalize
x_train=np.reshape(x_train,[-1,img_size,img_size,1])
x_test=np.reshape(x_test,[-1,img_size,img_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# In[7]:


# network parameters
input_shape=(img_size,img_size,1)
batch_size=32
kernel=3
droupout=.4
num_filters=32


# In[8]:


# left branch of Y network
left_inputs=Input(shape=input_shape)
x=left_inputs
filters=num_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    x=Conv2D(filters=filters,kernel_size=kernel,
            padding='same',activation='relu')(x)
    x=Dropout(droupout)(x)
    x=MaxPool2D()(x)
    filters*=2


# In[9]:


# right branch of Y network
right_inputs=Input(shape=input_shape)
y=right_inputs
filters=num_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    y=Conv2D(filters=filters,kernel_size=kernel,
            padding='same',activation='relu',dilation_rate=2)(y)
    y=Dropout(droupout)(y)
    y=MaxPool2D()(y)
    filters*=2


# In[10]:


# merge left and right branches outputs
y=concatenate([x,y])
# feature maps to vector before connecting to Dense 
y=Flatten()(y)
y=Dropout(droupout)(y)
outputs=Dense(nu_labels,activation='softmax')(y)


# In[11]:


# build the model in functional API
model=Model([left_inputs,right_inputs],outputs)


# In[12]:


# verify the model using graph
plot_model(model, to_file='cnn-y-network.png', show_shapes=True)
# verify the model using layer text description
model.summary()


# In[13]:


model.compile(loss='categorical_crossentropy',optimizer='adam',
             metrics=['accuracy'])


# In[14]:


# train the model with input images and labels
model.fit([x_train,x_train],y_train,
         validation_data=([x_test,x_test],y_test),epochs=20,batch_size=batch_size)


# In[15]:


# model accuracy on test dataset
score = model.evaluate([x_test, x_test],
 y_test,
 batch_size=batch_size,
 verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


# In[ ]:




