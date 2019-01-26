#coding:utf-8
"""
在model = models.EncodeProcessDecode(node_output_size=3)中实现下面这个网络结构
                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*

class EncodeProcessDecode(snt.AbstractModule):继承于snt.AbstractModule
首先我们来了解一下snt.AbstractModule的基本情况
文档：https://github.com/deepmind/sonnet/tree/master/docs
Sonnet是一个建立在TensorFlow之上的库，用于构建复杂的神经网络。

（1）
基本原则是分离了操作设计与连接，基本的使用方法是：
first construct Python objects which represent some part of a neural network,
and then separately connect these objects into the TensorFlow computation graph.
The objects are subclasses of sonnet.AbstractModule
定义的操作，可以反复使用，参数是共享的。比如可以输入训练数据和测试数据
The variables (i.e., the weights and biases of the linear transformation) are automatically shared.

（2）
python类的下划线
“单下划线开头” 的成员变量叫做保护变量，意思是只有类对象和子类对象才能访问到这些变量。
“双下划线开头” 的是私有成员，意思是只有类对象自己能访问，连子类对象也不能访问到这个数据。

（3）
构造函数的第一件事应该是调用超类构造函数，传入模块的名称，实现变量共享
super(EncodeProcessDecode, self).__init__(name=name)

（4）
需要实现_build()方法
This will be called whenever the module is connected into the tf.Graph
build方法的输入，可以是：（1）空的，（2）单个Tensor，（3）或包含多个Tensors的任意结构。
多个张量可以提供元组或命名元组，其元素又可以是张量或其他元组/命名元组。
大多数输入张量需要批量维度，如果张量具有颜色通道，则它必须是最后一个维度。
"""

