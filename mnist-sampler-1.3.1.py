#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# In[3]:


# load datasets
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:


# count the number of unique train labels
unique,counts=np.unique(y_train,return_counts=True)
print("train labels:",dict(zip(unique,counts)))


# In[5]:


# count the number of unique test labels
unique,counts=np.unique(y_test,return_counts=True)
print("test labels:",dict(zip(unique,counts)))


# In[6]:


# sample 25 mnist digits from train dataset
indeces=np.random.randint(0,x_train.shape[0],size=25)
images=x_train[indeces]
labels=y_train[indeces]


# In[7]:


# plot the 25 mnist digits
plt.figure(figsize=(5,5))
for i in range (len(indeces)):
    plt.subplot(5,5,i+1)
    image=images[i]
    plt.imshow(image,cmap='gray')
    plt.axis("off")


# In[8]:


plt.savefig("mnist-samples.png")
plt.show()
plt.close('all')


# In[ ]:




