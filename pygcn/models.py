import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)  # 第一层
        self.gc2 = GraphConvolution(nhid, nclass)  # 第二层
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # 进行非线性变化
        x = F.dropout(x, self.dropout, training=self.training)  # dropout自动断连一些
        x = self.gc2(x, adj)  # 第二层
        return F.log_softmax(x, dim=1)  # 返回概率
