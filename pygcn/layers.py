import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):  # in_features特征数 out_features输出的类别数
        super(GraphConvolution, self).__init__()  # 调用父类的构造函数已使用其成员函数
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 初始化权重矩阵和bias 范围为输出的1/根号输出维度
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # input(N, in_features) support(N, out_features)
        support = torch.mm(input, self.weight)  # (N, out_features)	线性变换后的特征矩阵
        output = torch.spmm(adj, support)  # (N, out_features)	邻居聚合后的输出特征 torch.spmm稀疏矩阵的乘法
        if self.bias is not None:
            return output + self.bias  # H' = Û · (H · W) + b
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'  # print(gcn_layer)  # 打印实例，会自动调用 __repr__ 方法
