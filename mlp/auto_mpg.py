
# coding: utf-8

# In[1]:


from __future__ import absolute_import,division,print_function

import seaborn as sns
import pathlib2
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# In[2]:


print(tf.__version__)


# In[3]:


dataset_path = '/home/wangluguang/lr/dataset/auto-mpg.data'


# In[4]:


column_names = ['MPG','Gang Shu','Pai Liang','Ma Li','Che Zhong','Jia Su','Chu Chang','Chan Di']
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values='?',comment='\t',sep=' ',skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.tail()


# In[5]:


len(dataset),len(raw_dataset)


# In[6]:


dataset.isna().sum()
dataset = dataset.dropna()


# In[7]:


len(dataset),len(raw_dataset)


# In[8]:


origin = dataset.pop('Chan Di')


# In[9]:


dataset['Mei Guo']=(origin==1)*1.0
dataset['Ou Zhou']=(origin==2)*1.0
dataset['Ri Ben']=(origin==3)*1.0


# In[10]:


dataset.tail()


# In[11]:


train_set = dataset.sample(frac=0.8,random_state=0)
test_set = dataset.drop(train_set.index)


# In[12]:


len(train_set),len(test_set)


# In[13]:


train_set.tail()


# In[14]:


sns.pairplot(train_set[['MPG','Che Zhong']],diag_kind='kde')


# In[15]:


train_desc = train_set.describe()
train_desc.pop('MPG')
train_desc = train_desc.transpose()


# In[16]:


train_label = train_set.pop('MPG')
test_label = test_set.pop('MPG')


# In[19]:


train_desc


# In[20]:


def norm(x):
    return (x-train_desc['mean'])/train_desc['std']

normed_train_data = norm(train_set)
normed_test_data = norm(test_set)


# In[24]:


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,input_shape=[(len(train_set.keys()))]),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    
    return model


# In[25]:


model = build_model()


# In[26]:


model.summary()


# In[29]:


example_batch = normed_train_data[:10]
model.predict(example_batch)


# In[35]:


history = model.fit(normed_train_data,train_label,epochs=1000,validation_split=0.2,verbose=0)


# In[36]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[38]:


import matplotlib.pyplot as plt

plt.figure()

