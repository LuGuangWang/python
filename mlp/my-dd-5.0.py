
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from xpinyin import Pinyin
import re
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import math


# In[2]:


print(tf.__version__)


# In[3]:


data_set = pd.read_excel('my_dataset _4.0.xlsx')
data_set = data_set.drop_duplicates()
data_set = data_set[:1600]
data_set = data_set.reset_index(drop=True)


# In[4]:


data_set.loc[1502]


# In[5]:


data_set.insert(2,2,data_set[0])


# In[6]:


def wordsToWord(p,words):
    if type(words)==int or type(words)==float:
        dst=str(words)
    else: 
        dst=words
       
    pinyin=p.get_pinyin(dst, ' ').lower()
    pinyin=re.sub(r'[^a-z0-9\s]',' ',pinyin).strip()
    pinyin=re.sub(r'\s+',' ',pinyin)
    tmp=pinyin.split(" ")
    res=[]
    m=re.compile(r'[a-z]+[0-9]+|[0-9]+[a-z]+')
    d=re.compile(r'\d+')
    s=re.compile(r'[a-z]+')
    for w in tmp:
        flag=m.match(w)
        if flag:
            while len(w) > 0:
                dig = d.match(w)
                ele = s.match(w)
                t = dig
                if t is None:
                    t = ele
                res.append(t.group())
                w=w.replace(t.group(),'',1)
        elif len(w)>0:
            res.append(w)
    return res


# In[7]:


def col_len(p,col):
    if type(col)==int or type(col)==float:
        dst=str(col)
    else: 
        dst=col
        
    pinyin=p.get_pinyin(dst, ' ')
    
    shibie=re.sub(r'[^a-z0-9\s]',' ',pinyin).strip()
    shibie=re.sub(r'\s+',' ',shibie)
    
    weishibie = re.sub(r'[a-z0-9\s]',' ',pinyin).strip()
    weishibie = re.sub(r'\s+',' ',weishibie)
    
    wordsCount = shibieCount = unCount =0
    
    if len(pinyin)>0:
        wordsCount = len(pinyin.split(' '))
    if len(shibie)>0:
        shibieCount = len(shibie.split(' '))
    if len(weishibie)>0:
        unCount = len(weishibie.split(' '))
    
    if shibieCount > wordsCount:
        shibieCount = wordsCount
        
    if unCount > wordsCount:
        unCount = wordsCount
    
    return wordsCount,shibieCount,unCount
    
    #return pinyin,shibie,weishibie


# In[8]:


p = Pinyin()
data_set[3] = data_set[0].map(lambda col: col_len(p,col))
data_set[2] = data_set[0].map(lambda col: wordsToWord(p,col))


# In[9]:


data_set.loc[1502]


# In[10]:


dic=set()
dig=re.compile(r'\d+')
max_len=0
for d in data_set[2]:
    max_len=max(len(d),max_len)
    for i in d:
        flag = dig.match(i)
        if flag:
            dic.add(str(len(i)))
        else:
            dic.add(i)
dic=list(dic)
dic.sort()


# In[11]:


#data_set[2] = data_set[2].map(lambda col: wordToVec(col))

#def wordToVec(words):
#    res = np.zeros(max_len,dtype=int)
#    i = 0
#    for w in words:
#        if w.isdigit():
#            res[i] = dic.index(len(w)) + 1
#        else:
#            res[i] = dic.index(w) + 1
#        i+=1
#    return res

def merge_col(df):
    extend = 0
    res = np.zeros(max_len+extend,dtype=int)
    #res[0] = df[3][0]
    #res[1] = df[3][1]
    #res[2] = df[3][2]
    i = extend
    for w in df[2]:
        if w.isdigit():
            res[i] = dic.index(str(len(w))) + 1
        else:
            res[i] = dic.index(w) + 1
        i+=1
    df[2] = res
    return df

data_set = data_set.apply(merge_col,axis=1)


# In[12]:


data_set.loc[1502]


# In[13]:


data_set = data_set.reindex(np.random.permutation(data_set.index))
data_set = data_set.reset_index(drop=True)

data_set.tail()


# In[14]:


num_classes = 2
max_len 

filer_height = 1
filter_width = 4
filter_channel = 1
out_channel = 1


# In[15]:


max_len,num_classes


# In[16]:


train_sor = tf.placeholder(dtype=tf.int32,shape=[None,1,max_len],name='train_data')
label = tf.placeholder(dtype=tf.float32,shape=[None,num_classes],name='train_label')

print('train_sor',train_sor)
print('label',label)

#size = train_sor.get_shape().as_list()[0]

# embedding weights
emb_w = tf.get_variable("emb_w", [3000,1])

#embedding layer
l1_emb = tf.nn.embedding_lookup(emb_w,train_sor)
l1_emb = tf.layers.batch_normalization(l1_emb,name='l1_emb')
print(l1_emb)

conv2d_shape = [filer_height,filter_width, filter_channel,out_channel]
pool_shape = [1,filer_height,filter_width,1]

print('conv2d_shape ',conv2d_shape)
print('pool_shape ',pool_shape)

#conv1d weights
conv2d_w = tf.Variable(tf.truncated_normal(conv2d_shape, stddev=0.1),name='conv2d_w')
#conv1d bias
conv2d_b = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="conv2d_b")
#conv1d layer
l2_conv2d = tf.nn.conv2d(input=l1_emb,filter=conv2d_w,strides=[1,1,1,1],padding='SAME',name='conv2d')
print(l2_conv2d)

#relu layer
l3_relu = tf.nn.relu(tf.nn.bias_add(l2_conv2d, conv2d_b),name='relu')
print(l3_relu)

#max pooling layer
l4_maxpool = tf.nn.max_pool(l3_relu,ksize=pool_shape,strides=[1,1,1,1],padding='SAME',name='max_pool')
l4_maxpool = tf.reshape(l4_maxpool,[-1,max_len])
print(l4_maxpool)

#dropout layer
l5_dropout = tf.nn.dropout(l4_maxpool,keep_prob=0.5)
print(l5_dropout)

#l2_loss = tf.constant(0.0)
W = tf.get_variable('W',shape=[max_len,num_classes],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
#l2_loss += tf.nn.l2_loss(W)
#l2_loss += tf.nn.l2_loss(b)
scores = tf.nn.xw_plus_b(l5_dropout, W, b, name="scores")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=label)

global_step = tf.Variable(0, name="global_step", trainable=False)

train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy,global_step=global_step)

#loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
train_loss = tf.reduce_mean(cross_entropy)

predictions = tf.argmax(scores, 1, name="predictions")
correct_predictions = tf.equal(predictions, tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


#loss_summary = tf.summary.scalar("loss", train_loss)
#acc_summary = tf.summary.scalar("accuracy", accuracy)
#train_summary_op = tf.summary.merge([loss_summary, acc_summary])


# In[17]:


train_size = math.ceil(len(data_set) * 0.7)
train_data = data_set[:train_size]
test_data = data_set[train_size:]

test_data[:10]


# In[18]:


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(1)
tf.set_random_seed(1)
path = 'dd_path/ckpt'

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    #train_summary_dir = os.path.join('dd_path', "summaries", "train")
    #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
    batch_size=300
    pages = math.ceil(1.0*len(train_data)/batch_size) + 1
    for _ in range(300):
        for page in range(1,pages):
            start = (page - 1) * batch_size
            end = min(page * batch_size,len(train_data))
            batch_data_set = train_data[start:end]
            batch_data_set = batch_data_set.reindex(np.random.permutation(batch_data_set.index))

            batch_train_set = np.array(batch_data_set[2].tolist())
            batch_train_set = batch_train_set[:,np.newaxis,:]

            batch_train_label = np.array(batch_data_set[1].tolist())
            train_pos_label = batch_train_label.reshape(len(batch_train_label),1)
            train_neg_label = 1 - train_pos_label
            batch_train_label = np.concatenate((train_pos_label,train_neg_label),axis=-1)
            step,loss,acc,_ = sess.run([global_step,train_loss,accuracy,train_op],feed_dict={train_sor:batch_train_set,label:batch_train_label})
            #step,loss,acc,_,summaries = sess.run([global_step,train_loss,accuracy,train_op,train_summary_op],feed_dict={train_sor:batch_train_set,label:batch_train_label})
            #train_summary_writer.add_summary(summaries, step)
            
            if step%50==0:
                valid_data = test_data.reindex(np.random.permutation(test_data.index))
                #valid_data = test_data
                test_set = np.array(valid_data[2].tolist())
                test_set = test_set[:,np.newaxis,:]

                test_label = np.array(valid_data[1].tolist())
                test_pos_label = test_label.reshape(len(test_label),1)
                test_neg_label = 1 - test_pos_label
                test_label = np.concatenate((test_pos_label,test_neg_label),axis=-1)

                v_loss,v_acc = sess.run([train_loss,accuracy],feed_dict={train_sor:test_set,label:test_label})
                print("step {}, loss {:g}, v_loss {:g}, acc {:g}, v_acc {:g}".format(step, loss,v_loss, acc,v_acc))
                saver = tf.train.Saver()
                saver.save(sess, save_path=path)
    
    print(sess.run(accuracy,feed_dict={train_sor:test_set,label:test_label}))


# In[29]:


test_set.shape


# In[31]:


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('dd_path/ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('dd_path'))

    print(sess.run(accuracy,feed_dict={train_sor:test_set,label:test_label}))

