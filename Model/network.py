from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
from torch.nn import Dropout, GRU, Linear, Tanh, Sigmoid
from torch.nn import Sequential, ModuleList

from Base import BaseModel

'''
SELDnet（Sound Event Localization and Detection Network）是一个用于同时检测和定位声音事件的深度学习模型。它结合了卷积神经网络（CNN）、循环神经网络（RNN）和全连接神经网络（FNN）等多种类型的神经网络。下面是该网络的详细架构：

    1. 输入层：SELDnet接收一个具有特定形状的张量，例如（通道数，频率，时间）。

    2. 2D卷积层序列：输入首先经过一系列的2D卷积层。在每个卷积层之后，都有一个批量归一化（BatchNorm）层和ReLU激活函数。然后通过一个最大池化层进行下采样，最后应用一个Dropout层。

    3. 双向RNN层：经过卷积层序列处理的张量经过调整形状后，被送入一个或多个双向GRU层。这些双向RNN层可以捕捉时间维度上的长时依赖关系。

    4. DOA（Direction of Arrival）分支：双向RNN层的输出被送入DOA分支，该分支包含一系列全连接层、Dropout层和激活函数（Tanh）层。DOA分支主要用于预测声源的方向。

    5. SED（Sound Event Detection）分支：双向RNN层的输出也被送入SED分支，该分支包含一系列全连接层、Dropout层和激活函数（Sigmoid）层。SED分支主要用于检测声音事件的存在。

网络的输出包括两个部分：SED分支的输出（表示声音事件的存在概率）和DOA分支的输出（表示声源方向）。

下面是SELDnet的网络结构示意图：

    Input
    |
    v
    CNN Layers -> BatchNorm -> ReLU -> MaxPooling -> Dropout
    |
    v
    Bi-directional RNN Layers (GRU)
    |
    v
    +--> DOA Branch (FNN) -> Dropout -> Tanh
    |
    +--> SED Branch (FNN) -> Dropout -> Sigmoid
'''


# 定义SELDnet类，继承PyTorch的Module类
class SELDnet(BaseModel):
    def __init__(
            self,
            data_in,  # 输入数据的形状
            data_out,  # 输出数据的形状
            dropout_rate,  # Dropout层的丢弃概率
            nb_cnn2d_filt,  # 2D卷积层的滤波器数量
            pool_size,  # 池化层的大小列表
            rnn_size,  # RNN层的隐藏单元数量列表
            fnn_size  # FNN层的神经元数量列表
    ):

        # 调用父类的构造函数
        super(SELDnet, self).__init__()

        # 创建2D卷积、批标准化、激活函数、池化和Dropout层序列
        self.cnn_layers = Sequential()
        for i, convCnt in enumerate(pool_size):
            self.cnn_layers.add_module(f'conv2d_{i}', Conv2d(data_in[-3], nb_cnn2d_filt, kernel_size=(3, 3), padding=1))
            self.cnn_layers.add_module(f'batchnorm_{i}', BatchNorm2d(nb_cnn2d_filt))
            self.cnn_layers.add_module(f'activation_{i}', ReLU())
            self.cnn_layers.add_module(f'maxpool2d_{i}', MaxPool2d(kernel_size=(1, pool_size[i])))
            self.cnn_layers.add_module(f'dropout_{i}', Dropout(dropout_rate))

        # 创建双向RNN层列表
        self.birnn_layers = ModuleList()
        for nb_rnn_filt in rnn_size:
            self.birnn_layers.append(GRU(nb_cnn2d_filt * data_in[-1], nb_rnn_filt, batch_first=True, bidirectional=True,
                                         dropout=dropout_rate))

        # 创建DOA分支的全连接、Dropout、激活函数层序列
        self.doa_layers = Sequential()
        for i, nb_fnn_filt in enumerate(fnn_size):
            self.doa_layers.add_module(f'doatime_dense_{i}', Linear(nb_rnn_filt * 2, nb_fnn_filt))
            self.doa_layers.add_module(f'doatime_dropout_{i}', Dropout(dropout_rate))
        self.doa_layers.add_module('doa_time_dense_final', Linear(nb_fnn_filt, data_out[1][-1]))
        self.doa_layers.add_module('doa_activation', Tanh())

        # 创建SED分支的全连接、Dropout、激活函数层序列
        self.sed_layers = Sequential()
        for i, nb_fnn_filt in enumerate(fnn_size):
            self.sed_layers.add_module(f'sedtime_dense_{i}', Linear(nb_rnn_filt * 2, nb_fnn_filt))
            self.sed_layers.add_module(f'sedtime_dropout_{i}', Dropout(dropout_rate))

        self.sed_layers.add_module('sed_time_dense_final', Linear(nb_fnn_filt, data_out[0][-1]))
        self.sed_layers.add_module('sed_activation', Sigmoid())

    # 定义前向传播过程
    def forward(self, x):
        # 将输入传递给卷积层序列
        x = self.cnn_layers(x)

        # 调整张量的维度，以便将其输入到RNN层
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        # 通过双向RNN层传递张量
        for birnn in self.birnn_layers:
            x, _ = birnn(x)

        # 将张量传递给DOA分支
        doa = self.doa_layers(x)

        # 将张量传递给SED分支
        sed = self.sed_layers(x)

        # 返回SED和DOA分支的输出
        return sed, doa
