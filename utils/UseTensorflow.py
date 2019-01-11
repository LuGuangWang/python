import tensorflow as tf

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t0=[1,2,3,4,5]
tt=[6,7,8,9,10]

#数组运算
tmp1=tf.convert_to_tensor(t1,dtype=tf.int32)
tmp2=tf.convert_to_tensor(t2,dtype=tf.int32)
tmp3=tmp2*tmp1

#拼接
a = tf.placeholder(dtype=tf.int32, shape=None, name='a')
b = tf.placeholder(dtype=tf.int32, shape=None, name='b')
t3 = tf.concat([a, b], axis=1, name="concat")
t11=tf.concat([t1,t2],axis=-1)

a1 = tf.constant(t1,name='a1')
b1 = tf.constant(t2,name='b2')
t4 = tf.stack([a1,b1],axis=1,name="stack")

#抽取
t5=tf.slice(t1,[0,0],[2,2],name="slice")
t6=tf.gather(t0,1,axis=0,name='gather')
t7=tf.nn.embedding_lookup(t1,[[[1],[2]]],name='embedding')

#复制
t8 = tf.tile(t1,[2,1])

#变量
c=tf.Variable(2,name="c")
d=tf.Variable(3,name='d')
d=tf.assign(d,10)
t10 = tf.add(c,d,name='add')

with tf.Session() as sess:
    #tf.summary.FileWriter("../logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    print('=====t3:\n',sess.run(t3,feed_dict={a:t1,b:t2}))
    print('=====t4:\n',sess.run(t4))
    print('=====t5:\n',sess.run(t5))
    print('=====t6:\n', sess.run(t6))
    print('=====t7:\n', sess.run(t7))
    print('=====t8:\n', sess.run(t8))
    print('=====t11:\n', sess.run(t11))

    print('=====tmp3:\n',sess.run(tmp3))
