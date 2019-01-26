#coding:utf-8
"""
data_dicts_to_graphs_tuple->
_to_compatible_data_dicts->
tf.convert_to_tensor

This function can be useful when composing a new operation in Python (such as my_func in the example above).
All standard Python op constructors apply this function to each of their Tensor-valued inputs,
which allows those ops to accept numpy arrays, Python lists, and scalars in addition to Tensor objects.
在_to_compatible_data_dicts(data_dicts):函数中传入Python lists组织的图数据结构，变成tensor给图网络使用
"""

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
"""

"""
data_dicts_to_graphs_tuple->
_concatenate_data_dicts->  上一步用tf.convert_to_tensor生成了 a list of dict 现在转换成 a dict of lists 然后拼接起来
dct = collections.defaultdict(lambda: [])     

(1)Python中的lambda

在Python中，lambda的语法是唯一的。其形式如下：
 lambda argument_list: expression
其中，lambda是Python预留的关键字，argument_list和expression由用户自定义。

lambda语法是固定的，其本质上只有一种用法，那就是定义一个lambda函数。在实际中，根据这个lambda函数应用场景的不同，可以将lambda函数的用法扩展为以下几种：
(1.1)变量赋值
将lambda函数赋值给一个变量，通过这个变量间接调用该lambda函数。
例如，执行语句add=lambda x, y: x+y，定义了加法函数lambda x, y: x+y，并将其赋值给变量add，这样变量add便成为具有加法功能的函数。
例如，执行add(1,2)，输出为3。
time.sleep=lambda x:None。这样，在后续代码中调用time库的sleep函数将不会执行原有的功能。例如，执行time.sleep(3)时，程序不会休眠3秒钟

(1.2)参数传递
部分Python内置函数接收函数作为参数。
filter函数。此时lambda函数用于指定过滤列表元素的条件。
例如filter(lambda x: x % 3 == 0, [1, 2, 3])指定将列表[1,2,3]中能够被3整除的元素过滤出来，其结果是[3]。

(2)python中的collections.defaultdict

python中通过Key访问字典，当Key不存在时，会引发‘KeyError’异常。为了避免这种情况的发生，
可以使用collections类中的defaultdict()方法来为字典提供默认值。
语法格式： 
collections.defaultdict([default_factory[, …]])
该函数返回一个类似字典的对象。defaultdict是Python内建字典类（dict）的一个子类，它重写了方法_missing_(key)，
增加了一个可写的实例变量default_factory,实例变量default_factory被missing()方法使用

"""

"""
data_dicts_to_graphs_tuple->
graphs.GraphsTuple 将拼接好的 a dict of lists 传入，生成最终的 GraphsTuple 对象

(1)调用时 ** 把字典变成参数
Python中*args和**kwargs 是处理可变长参数，前者处理成tuple后者处理成字典

(2)调用超类的构造方法
class A(object):
  def __init__(self):
    self.nameaa = 'aa'

  def funca(self):
    print('function a %s' % self.nameaa)


class B(A):
  def __init__(self):
    self.namebb = 'bb'
    super(B, self).__init__()

  def funcb(self):
    print('function b %s' % self.namebb)


b = B()
print(b.namebb)
b.funcb()
print(b.nameaa)
b.funca()


(3)namedtuple对象就如它的名字说定义的那样，你可以给tuple命名

"""
import collections

Person = collections.namedtuple('Person', 'name age gender')
print('Type of Person:', type(Person))
Bob = Person(name='Bob', age=30, gender='male')
print('Representation:', Bob)
Jane = Person(name='Jane', age=29, gender='female')
print('Field by Name:', Jane.name)
for people in [Bob, Jane]:
  print("%s is %d years old %s" % people)


NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"
N_NODE = "n_node"
N_EDGE = "n_edge"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS)
GRAPH_NUMBER_FIELDS = (N_NODE, N_EDGE)
ALL_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE)


class GraphsTuple(
    collections.namedtuple("GraphsTuple",
                           GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS)):
  def __init__(self, *args, **kwargs):
    del args, kwargs
    # The fields of a `namedtuple` are filled in the `__new__` method.
    # `__init__` does not accept parameters.
    super(GraphsTuple, self).__init__()

GraphsTuple(**dict(nodes=1,edges=2,receivers=3,senders=4,globals=5,n_node=6,n_edge=7))