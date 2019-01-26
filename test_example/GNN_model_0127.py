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

*（3）构造函数中做的是一些参数设置操作
构造函数的第一件事应该是调用超类构造函数，传入模块的名称，实现变量共享
super(EncodeProcessDecode, self).__init__(name=name)

*（4）这里是定义图操作的地方
需要实现_build()方法

build方法的输入——sonnet模块区分了数据链接和数据操作，build的参数中会有input选项，满足规范的input会输入给model中定义的操作
  def _build(self, inputs):
    lin_x_to_h = snt.Linear(output_size=self._hidden_size, name="x_to_h")
    lin_h_to_o = snt.Linear(output_size=self._output_size, name="h_to_o")
    return lin_h_to_o(self._nonlinearity(lin_x_to_h(inputs)))
    #反正每个model输入一个tensor输出一个tensor，落在底层的sonnet操作的tensor应该是单个tensor
    #如果model有很多层次结构的话，可以用 namedtuple，然后在build中拆开

This will be called whenever the module is connected into the tf.Graph
build方法的输入，可以是：（1）空的，（2）单个Tensor，（3）或包含多个Tensors的任意结构。
多个张量可以提供元组或命名元组，其元素又可以是张量或其他元组/命名元组。
大多数输入张量需要批量维度，如果张量具有颜色通道，则它必须是最后一个维度。

build()方法常用来：
构建和使用内部模块
使用已存在的模块，并将其传递给构造函数
直接创建变量（只能用tf.get_variable以便能够复用）。
"""

"""
模块之间的相互关系：

EncodeProcessDecode
    self._encoder = MLPGraphIndependent()
        self._network = modules.GraphIndependent(   
              edge_model_fn=make_mlp_model,         #sonnet的snt.Sequential是什么
              node_model_fn=make_mlp_model,
              global_model_fn=make_mlp_model)
              
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
"""
