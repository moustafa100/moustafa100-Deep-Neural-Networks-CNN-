#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import mnist
import graphviz
import pydot


# In[9]:


# load datasets (mnist)
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[10]:


#compute the number of labels
nu_labels=len(np.unique(y_train))


# In[11]:


#convert into one_hot vector
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[12]:


# input image dimensions
img_size=x_train.shape[1]


# In[13]:


# resize and normalize
x_train=np.reshape(x_train,[-1,img_size,img_size,1])
x_test=np.reshape(x_test,[-1,img_size,img_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# In[7]:


# network parameters
# image is processed as is (square grayscale)
input_shape=(img_size,img_size,1)
kernel=3
pool=2
batch=128
fillter=64
droupout=.2


# In[16]:


# model is a stack of CNN-ReLU-MaxPooling
model=Sequential()
model.add(Conv2D(filters=fillter,kernel_size=kernel,
                activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool))
model.add(Conv2D(filters=fillter,kernel_size=kernel,
                activation='relu'))
model.add(MaxPooling2D(pool))
model.add(Flatten())
# dropout added as regularizer
model.add(Dropout(droupout))
# output layer is 10-dim one-hot vector
model.add(Dense(nu_labels))
model.add(Activation('softmax'))


# In[18]:


model.summary()


# In[19]:


# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])


# In[20]:


# train the network
model.fit(x_train,y_train,epochs=10,batch_size=batch)


# In[21]:


_,acc=model.evaluate(x_test,y_test,
                    batch_size=batch,verbose=0)
print("\n Test Accuracy:%.1f%%"%(acc*100.0))


# In[ ]:




