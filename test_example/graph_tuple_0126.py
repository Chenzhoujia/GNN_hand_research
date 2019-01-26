#coding:utf-8
"""
This function can be useful when composing a new operation in Python (such as my_func in the example above).
All standard Python op constructors apply this function to each of their Tensor-valued inputs,
which allows those ops to accept numpy arrays, Python lists, and scalars in addition to Tensor objects.
在_to_compatible_data_dicts(data_dicts):函数中传入Python lists组织的图数据结构，变成tensor给图网络使用
"""

import numpy as np
import tensorflow as tf
def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))