from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from pygcn.utils import encode_onehot, accuracy, sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN
import scipy.sparse as sp


# 设置随机种子以保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# 加载数据（支持不同归一化方式）
def load_data_ablation(path="./data/cora/", dataset="cora", add_self_loop=True, normalization='left'):
    """
    Load citation network dataset with ablation options
    
    Args:
        add_self_loop: 是否添加自环
        normalization: 'left' (左归一化 D^-1 A), 
                       'symmetric' (对称归一化 D^-0.5 A D^-0.5),
                       'none' (不归一化)
    """
    print(f'Loading {dataset} dataset... (self_loop={add_self_loop}, normalization={normalization})')

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 特征归一化 (行归一化)
    features = normalize_features(features)
    
    # 根据配置处理邻接矩阵
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0])
    
    # 邻接矩阵归一化
    if normalization == 'symmetric':
        adj = normalize_adj_symmetric(adj)
    elif normalization == 'left':
        adj = normalize_adj_left(adj)
    elif normalization == 'none':
        pass  # 不归一化
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(mx):
    """Row-normalize sparse matrix (用于特征)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj_symmetric(adj):
    """
    对称归一化: D^-0.5 A D^-0.5
    这是GCN论文中使用的标准归一化方式
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # D^-0.5 A D^-0.5
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_left(adj):
    """
    左归一化: D^-1 A
    仅对行进行归一化
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).tocoo()


class GCNTrainer:
    """GCN训练器，支持消融实验"""
    
    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test, 
                 hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4, 
                 epochs=200, early_stopping=None, cuda=False):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.cuda = cuda
        
        # 模型
        self.model = GCN(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=labels.max().item() + 1,
                        dropout=dropout)
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        
        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
        return loss_train.item(), acc_train.item(), time.time() - t
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.features, self.adj)
            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
            acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        return loss_val.item(), acc_val.item()
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.features, self.adj)
            loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
            acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        return loss_test.item(), acc_test.item()
    
    def fit(self, verbose=True):
        """训练模型，使用tqdm显示进度"""
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        pbar = tqdm(range(self.epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            loss_train, acc_train, train_time = self.train_epoch()
            loss_val, acc_val = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(loss_train)
            self.history['train_acc'].append(acc_train)
            self.history['val_loss'].append(loss_val)
            self.history['val_acc'].append(acc_val)
            
            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{loss_train:.4f}',
                'train_acc': f'{acc_train:.4f}',
                'val_acc': f'{acc_val:.4f}'
            })
            
            # Early stopping
            if self.early_stopping is not None:
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_epoch = epoch
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.early_stopping:
                    if verbose:
                        pbar.write(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                # 没有early stopping，记录最后一轮
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_epoch = epoch
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            if self.cuda:
                self.model.cuda()
        
        # 测试
        loss_test, acc_test = self.test()
        
        return {
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'test_loss': loss_test,
            'test_acc': acc_test,
            'history': self.history
        }


def run_ablation_study(seed=42, cuda=False):
    """运行消融实验
    
    基线: 对称归一化 + self-loop (GCN论文标准设置)
    消融对比:
    1. 对称归一化 + 无self-loop (测试self-loop的影响)
    2. 左归一化 + self-loop (测试归一化方式的影响)
    """
    
    # 统一的训练配置（与原始代码一致）
    config = {
        'hidden': 16,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs': 200,
        'early_stopping': None  # 不使用early stopping，与原始代码一致
    }
    
    print("=" * 80)
    print("GCN消融实验")
    print("=" * 80)
    print(f"训练配置: {config}")
    print("=" * 80)
    
    results = {}
    
    # 实验1: 基线 - 对称归一化 + self-loop (GCN论文标准)
    print("\n[实验1] 基线 - 对称归一化 + self-loop (D^-0.5 A D^-0.5)")
    print("-" * 80)
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data_ablation(
        add_self_loop=True, normalization='symmetric'
    )
    trainer = GCNTrainer(adj, features, labels, idx_train, idx_val, idx_test,
                        cuda=cuda, **config)
    results['baseline'] = trainer.fit(verbose=True)
    print(f"测试结果 - Loss: {results['baseline']['test_loss']:.4f}, "
          f"Accuracy: {results['baseline']['test_acc']:.4f}")
    
    # 实验2: 对称归一化 + 无self-loop (消融self-loop)
    print("\n[实验2] 对称归一化 + 无self-loop (消融self-loop)")
    print("-" * 80)
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data_ablation(
        add_self_loop=False, normalization='symmetric'
    )
    trainer = GCNTrainer(adj, features, labels, idx_train, idx_val, idx_test,
                        cuda=cuda, **config)
    results['no_self_loop'] = trainer.fit(verbose=True)
    print(f"测试结果 - Loss: {results['no_self_loop']['test_loss']:.4f}, "
          f"Accuracy: {results['no_self_loop']['test_acc']:.4f}")
    
    # 实验3: 左归一化 + self-loop (消融归一化方式)
    print("\n[实验3] 左归一化 + self-loop (消融归一化方式)")
    print("-" * 80)
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data_ablation(
        add_self_loop=True, normalization='left'
    )
    trainer = GCNTrainer(adj, features, labels, idx_train, idx_val, idx_test,
                        cuda=cuda, **config)
    results['left_norm'] = trainer.fit(verbose=True)
    print(f"测试结果 - Loss: {results['left_norm']['test_loss']:.4f}, "
          f"Accuracy: {results['left_norm']['test_acc']:.4f}")
    
    # 实验4: 有self-loop + 无归一化 (额外对比)
    print("\n[实验4] 无归一化 + self-loop (额外对比)")
    print("-" * 80)
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data_ablation(
        add_self_loop=True, normalization='none'
    )
    trainer = GCNTrainer(adj, features, labels, idx_train, idx_val, idx_test,
                        cuda=cuda, **config)
    results['no_norm'] = trainer.fit(verbose=True)
    print(f"测试结果 - Loss: {results['no_norm']['test_loss']:.4f}, "
          f"Accuracy: {results['no_norm']['test_acc']:.4f}")
    
    return results, config


def plot_results(results, save_path='ablation_curves.png'):
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'baseline': '#1f77b4',
        'no_self_loop': '#ff7f0e',
        'left_norm': '#2ca02c',
        'no_norm': '#d62728'
    }
    
    labels = {
        'baseline': 'Baseline (symmetric + self-loop)',
        'no_self_loop': 'No self-loop (symmetric)',
        'left_norm': 'Left norm (+self-loop)',
        'no_norm': 'No normalization (+self-loop)'
    }
    
    # 训练损失
    ax = axes[0, 0]
    for key in results:
        epochs = range(1, len(results[key]['history']['train_loss']) + 1)
        ax.plot(epochs, results[key]['history']['train_loss'], 
                color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 训练准确率
    ax = axes[0, 1]
    for key in results:
        epochs = range(1, len(results[key]['history']['train_acc']) + 1)
        ax.plot(epochs, results[key]['history']['train_acc'], 
                color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证损失
    ax = axes[1, 0]
    for key in results:
        epochs = range(1, len(results[key]['history']['val_loss']) + 1)
        ax.plot(epochs, results[key]['history']['val_loss'], 
                color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证准确率
    ax = axes[1, 1]
    for key in results:
        epochs = range(1, len(results[key]['history']['val_acc']) + 1)
        ax.plot(epochs, results[key]['history']['val_acc'], 
                color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存至: {save_path}")
    plt.close()


def print_summary_table(results, config):
    """打印消融实验总结表格"""
    print("\n" + "=" * 100)
    print("GCN消融实验总结")
    print("=" * 100)
    print(f"基线: 对称归一化 (D^-0.5 A D^-0.5) + Self-loop")
    print(f"训练配置: hidden={config['hidden']}, dropout={config['dropout']}, "
          f"lr={config['lr']}, weight_decay={config['weight_decay']}, "
          f"epochs={config['epochs']}, early_stopping={config['early_stopping']}")
    print("-" * 100)
    
    # 表头
    print(f"{'实验设置':<45} {'Best Epoch':<12} {'Val Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-" * 100)
    
    # 数据行
    exp_names = {
        'baseline': 'Baseline (symmetric + self-loop)',
        'no_self_loop': 'No self-loop (symmetric)',
        'left_norm': 'Left norm (+self-loop)',
        'no_norm': 'No normalization (+self-loop)'
    }
    
    for key in ['baseline', 'no_self_loop', 'left_norm', 'no_norm']:
        r = results[key]
        marker = " ★" if key == 'baseline' else ""
        print(f"{exp_names[key]:<45} {r['best_epoch']+1:<12} "
              f"{r['best_val_acc']:<12.4f} {r['test_loss']:<12.4f} {r['test_acc']:<12.4f}{marker}")
    
    print("-" * 100)
    
    # 对比分析
    baseline_acc = results['baseline']['test_acc']
    print("\n对比分析 (相对于基线):")
    print("-" * 100)
    
    for key in ['no_self_loop', 'left_norm', 'no_norm']:
        diff = results[key]['test_acc'] - baseline_acc
        diff_pct = (diff / baseline_acc) * 100 if baseline_acc > 0 else 0
        print(f"{exp_names[key]:<45} 测试准确率变化: {diff:+.4f} ({diff_pct:+.2f}%)")
    
    print("=" * 100)
    
    # Self-loop消融分析
    print("\n【Self-loop消融分析】")
    print("-" * 100)
    self_loop_diff = results['baseline']['test_acc'] - results['no_self_loop']['test_acc']
    print(f"添加self-loop带来的提升: {self_loop_diff:+.4f}")
    if self_loop_diff > 0:
        print("  → Self-loop有助于模型性能提升")
        print(f"    原因: Self-loop允许节点保留自身特征，避免过平滑")
    elif self_loop_diff < -0.01:
        print("  → Self-loop对性能有负面影响")
    else:
        print("  → Self-loop对性能影响不明显")
    
    # 归一化方式消融分析
    print("\n【归一化方式消融分析】")
    print("-" * 100)
    sym_vs_left = results['baseline']['test_acc'] - results['left_norm']['test_acc']
    sym_vs_none = results['baseline']['test_acc'] - results['no_norm']['test_acc']
    print(f"对称归一化 vs 左归一化: {sym_vs_left:+.4f}")
    print(f"对称归一化 vs 无归一化: {sym_vs_none:+.4f}")
    
    if sym_vs_left > 0:
        print("  → 对称归一化 (D^-0.5 A D^-0.5) 优于左归一化 (D^-1 A)")
        print("    原因: 对称归一化同时考虑入度和出度，更平衡")
    elif sym_vs_left < 0:
        print("  → 左归一化 (D^-1 A) 优于对称归一化 (D^-0.5 A D^-0.5)")
    else:
        print("  → 两种归一化方式效果相当")
        
    if sym_vs_none > 0:
        print("  → 归一化操作对模型性能至关重要")
        print("    原因: 归一化防止数值爆炸/消失，稳定训练")
    elif sym_vs_none < 0:
        print("  → 在此配置下，不归一化效果更好")
    else:
        print("  → 归一化操作对性能提升有限")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save-results', type=str, default='ablation_results.pkl',
                        help='Path to save results.')
    parser.add_argument('--save-plot', type=str, default='ablation_curves.png',
                        help='Path to save plot.')
    args = parser.parse_args()
    
    cuda = not args.no_cuda and torch.cuda.is_available()
    
    # 运行消融实验
    results, config = run_ablation_study(seed=args.seed, cuda=cuda)
    
    # 打印总结表格
    print_summary_table(results, config)
    
    # 绘制训练曲线
    plot_results(results, save_path=args.save_plot)
    
    # 保存结果
    with open(args.save_results, 'wb') as f:
        pickle.dump({'results': results, 'config': config}, f)
    print(f"实验结果已保存至: {args.save_results}")


if __name__ == '__main__':
    main()
