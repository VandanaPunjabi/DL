#!/usr/bin/env python
# coding: utf-8

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random


# In[42]:


df = pd.read_csv('DL-datasets/MNIST/mnist_784_csv.csv')


# In[43]:


df


# In[44]:


df.head()


# In[7]:


x = df.drop(['class'], axis = 1)
y = df['class'].values


# In[ ]:





# In[34]:


len(df)


# In[9]:


y


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[12]:


x_train.shape


# In[13]:


x_test.shape


# In[14]:


x_train = x_train/255


# In[15]:


x_test = x_test/255


# In[18]:


image_shape = (28,28)


# In[21]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[22]:


model.compile(optimizer='sgd',
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[25]:


#train usingthe fit
histroy = model.fit(x_train,y_train, epochs=10)


# In[26]:


test_loss , test_accuracy = model.evaluate(x_test,y_test)


# In[27]:


x_test_img = x_test.to_numpy().reshape(x_test.shape[0],28,28)


# In[31]:


plt.imshow(x_test_img[0])


# In[30]:


n = random.randint(0,500)
plt.imshow(x_test_img[n])
predicted_value = model.predict(x_test)
print(np.argmax(predicted_value[n]))


# In[33]:


plt.plot(histroy.history['loss'], label='loss', color='g')
plt.plot(histroy.history['accuracy'],label='accuracy', color='b')
plt.legend()
plt.show()


# In[ ]:




