import tensorflow as tf
import numpy as np
x = np.asarray([[1,2,3],[4,5,6],[41,51,61],[42,52,62],[41,51,61],[42,52,62]])
x_p = tf.placeholder(tf.int32,[6,3])
y =  tf.reshape(x, [3,2,3])
#y =  tf.reduce_sum(x_p,-1)
with tf.Session() as sess:
    y = sess.run(y,feed_dict={x_p:x})
    print y
