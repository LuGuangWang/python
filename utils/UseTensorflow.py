import tensorflow as tf

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]



#拼接
a = tf.placeholder(dtype=tf.int32, shape=None, name='a')
b = tf.placeholder(dtype=tf.int32, shape=None, name='b')
t3 = tf.concat([a, b], axis=-1, name="concat")

a1 = tf.constant(t1,name='a1')
b1 = tf.constant(t2,name='b2')
t4 = tf.stack([a1,b1],axis=1,name="stack")

#抽取
t5=tf.slice(t1,[0,0],[2,2],name="slice")
t6=tf.gather(t2,[0,1,2],axis=1,name='gather')

#变量
c=tf.Variable(2,name="c")
d=tf.Variable(3,name='d')
d=tf.assign(d,10)
t10 = tf.add(c,d,name='add')

with tf.Session() as sess:
    tf.summary.FileWriter("../logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    print('=====t3:\n',sess.run(t3,feed_dict={a:t1,b:t2}))
    print('=====t4:\n',sess.run(t4))
    print('=====t5:\n',sess.run(t5))
    print('=====t6:\n', sess.run(t6))

    print(sess.run(t10))
