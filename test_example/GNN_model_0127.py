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
需要实现_build()方法，build的参数中会有input选项，满足规范的input会输入给model中定义的操作

build方法的输入——sonnet模块区分了数据链接和数据操作，
  def _build(self, inputs):
    lin_x_to_h = snt.Linear(output_size=self._hidden_size, name="x_to_h")
    lin_h_to_o = snt.Linear(output_size=self._output_size, name="h_to_o")
    return lin_h_to_o(self._nonlinearity(lin_x_to_h(inputs)))
    #反正每个model输入一个tensor输出一个tensor，落在底层的sonnet操作的tensor应该是单个tensor
    #如果model有很多层次结构的话，可以用 namedtuple，然后在中间模块的build中拆开分发下去

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
              edge_model_fn=make_mlp_model,         #sonnet的snt.Sequential定义了一系列顺序操作，开头是build的input参数
                                                    # Sequential is a module which applies a number of inner modules or ops in
                                                    # sequence to the provided data. 
              node_model_fn=make_mlp_model,
              global_model_fn=make_mlp_model)
              
    self._core = MLPGraphNetwork()                  #这里实现了论文中的图网络计算流程
          self._network = modules.GraphNetwork(     #也是snt.AbstractModule的子类，接下来分别讲解构造函数和build函数中的内容
          make_mlp_model, make_mlp_model,make_mlp_model)
                                                    function GraphNetwork(E, V , u)
                                                        for k ∈ {1 . . . Ne} do
                                                            ek′ ← φe (ek, vrk, vsk, u) ⊲ 1. 计算更新边的属性
                                                        end for
                                                        for i ∈ {1 . . . Nn} do
                                                            let Ei′ = {(ek′, rk, sk)}rk=i, k=1:Ne
                                                            ¯ei′ ← ρe→v (Ei′)          ⊲ 2.　聚合每个点的相邻边
                                                            vi′ ← φv (¯ei′, vi, u)     ⊲ 3. 计算更新每个点的属性
                                                        end for
                                                        let V ′ = {v′}i=1:Nv
                                                        let E′ = {(ek′, rk, sk)}k=1:Ne
                                                        ¯e′ ← ρe→u (E′)                ⊲ 4. 聚合每个边
                                                        v¯′ ← ρv→u (V ′)               ⊲ 5. 聚合每个点
                                                        u′ ← φu (¯e′, v¯′, u)          ⊲ 6. 计算更新全局的属性
                                                        return (E′, V ′, u′)
                                                    end function
          
          (1)构造函数
          构造函数中传入了三个snt.Sequential定义的MLP
              分别传递给EdgeBlock以执行的可调用对象每边计算。给NodeBlock执行的可调用对象每节点计算。 给GlobalBlock以执行的可调用对象全局计算。
          构造函数还有４个参数，这些定义了聚合与更新操作中使用的参数——见https://arxiv.org/abs/1806.01261　图４
              reducer=tf.unsorted_segment_sum,——EdgeBlock和GlobalBlock中使用的聚合方法，传递给node_block_opt和global_block_opt
              edge_block_opt=None,——决定EdgeBlock中使用那些参数
              node_block_opt=None,——决定NodeBlock中使用那些参数
              global_block_opt=None，——决定GlobalBlock中使用那些参数
        　接下来，构造函数用上面三个MLP和相应配置，写self._edge_block，self._node_block，self._global_block
          传入的都是graph,在build函数中做了拆分，分别使用了边、点、全局属性
          
          (2)build函数——这个函数会在调用而非创建时使用，用来将操作连接到tensorflow的图中——output_ops_tr = model(graph_tr, num_processing_steps_tr)
          
          编码器是如何处理输入的？
            分别编码，最终得到的与输入相同也是一个Tuple，不过维度变得更高了，这时是没有链接的
            肯定是区分batch的（），按照例子看batch_size是shape=[batch_size, n_latent])
          
    self._decoder = MLPGraphIndependent()
"""
