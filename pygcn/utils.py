import numpy as np  # 主要进行计算
import scipy.sparse as sp  # 稀疏矩阵的运算
import torch  # 进行机器学习的主要库，包括卷积层的构建/一些参数的定义/自动求导/GPU上运行


def encode_onehot(labels):  # 将类别标签转换成编码 返回二维[0,0,1,0][1,0,0,0][0,0,1,0]....
    classes = set(labels)  # 集合
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}  # 生成每个标签对应的向量
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # 生成该标签的整体标签向量列表
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))  # 用于从文本文件中读取数据并生成数组
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 创建稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])  # 所有的类型生成向量组成矩阵

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 第一列（论文代码）生成列表
    idx_map = {j: i for i, j in enumerate(idx)}  # 生成论文代码和索引
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)  # [35,1033],[35,1100]...
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)  # [1,200],[1,20]...
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)  # 创建坐标格式的稀疏矩阵(edges[:, 0]：x，edges[:, 1]：y
    # np.ones(edges.shape[0])有链接就是1（有边） 这是有向图
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj.T > adj生成布尔矩阵 adj.T.multiply(adj.T > adj)得到  adj.multiply(adj.T > adj)好像没实际的作用
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # 节点特征归一化 A˜= A+ IN

    idx_train = range(140)  # 训练集的标签
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))  # 只是转换成张量形式（变成稠密矩阵）
    labels = torch.LongTensor(np.where(labels)[1])  # 取出每个文件的标签组成张量[1,2,3,1,2,5...]
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # 只是转换成张量形式（仍然为稀疏形式）

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)  # 从range变成张量形式

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):  # 归一化矩阵
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # (N, 1) 的 NumPy 数组（二维数组，每行一个元素）
    r_inv = np.power(rowsum, -1).flatten()  # 变成每行和的倒数的一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 将无穷大的数值变成0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):  # 用于计算模型的准确率
    preds = output.max(1)[1].type_as(labels)  # 找出预测的最高概率的类型
    correct = preds.eq(labels).double()  # 与labels标签比对正确为1.0 错为0.0
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 都转换成稀疏矩阵的坐标表示法
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64变成第一行表示行，第二行表示列
    # torch.from_numpy变成张量
    values = torch.from_numpy(sparse_mx.data)  # 同理将值也转换成张量
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    # tensor(indices=tensor([[0, 1, 1],
    #                        [1, 0, 2]]),
    #        values=tensor([1.0000, 1.0000, 1.0000]),
    #        shape=torch.Size([3, 3]), dtype=torch.float32,
    #        layout=torch.sparse_coo)
