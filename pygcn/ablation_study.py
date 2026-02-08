from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, normalize, sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--ablation', type=str, default='self_loop',
                    help='Ablation study type: self_loop or normalization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def load_data_with_config(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/cora/"), dataset="cora", add_self_loop=True, normalization="symmetric"):
    """Load citation network dataset with configurable self-loop and normalization"""
    print('Loading {} dataset...'.format(dataset))

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

    features = normalize(features)
    
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0])
    
    if normalization == "symmetric":
        # Symmetric normalization
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    elif normalization == "left":
        # Left normalization (row-wise)
        adj = normalize(adj)
    
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

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def train_model(adj, features, labels, idx_train, idx_val, idx_test, args):
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Training history
    train_history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        train_history['loss'].append(loss_train.item())
        train_history['acc'].append(acc_train.item())
        train_history['val_loss'].append(loss_val.item())
        train_history['val_acc'].append(acc_val.item())

        return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()

    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test.item(), acc_test.item()

    # Train model with tqdm
    print("\nTraining model...")
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        train(epoch)

    # Testing
    loss_test, acc_test = test()
    print(f"\nTest set results: loss= {loss_test:.4f}, accuracy= {acc_test:.4f}")

    return train_history, acc_test

def plot_training_curves(histories, labels, title, save_path):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history['acc'], label=f'{labels[i]} - Train Acc')
        plt.plot(history['val_acc'], label=f'{labels[i]} - Val Acc', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history['loss'], label=f'{labels[i]} - Train Loss')
        plt.plot(history['val_loss'], label=f'{labels[i]} - Val Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nTraining curves saved to {save_path}")

def main():
    if args.ablation == 'self_loop':
        print("=== Ablation Study: Self-Loop Effect ===")
        
        # Load data with self-loop (baseline)
        adj_with_loop, features, labels, idx_train, idx_val, idx_test = load_data_with_config(
            add_self_loop=True, normalization="symmetric"
        )
        
        # Load data without self-loop
        adj_without_loop, _, _, _, _, _ = load_data_with_config(
            add_self_loop=False, normalization="symmetric"
        )
        
        # Train with self-loop
        print("\n1. Training with self-loop:")
        history_with_loop, acc_with_loop = train_model(adj_with_loop, features, labels, idx_train, idx_val, idx_test, args)
        
        # Train without self-loop
        print("\n2. Training without self-loop:")
        history_without_loop, acc_without_loop = train_model(adj_without_loop, features, labels, idx_train, idx_val, idx_test, args)
        
        # Plot results
        plot_training_curves(
            [history_with_loop, history_without_loop],
            ['With Self-Loop', 'Without Self-Loop'],
            'Self-Loop Ablation Study',
            './results/self_loop_ablation.png'
        )
        
        # Save results
        results = pd.DataFrame({
            'Configuration': ['With Self-Loop', 'Without Self-Loop'],
            'Test Accuracy': [acc_with_loop, acc_without_loop]
        })
        results.to_csv('./results/self_loop_results.csv', index=False)
        
        print("\n=== Self-Loop Ablation Results ===")
        print(results)
        print(f"\nAccuracy difference: {abs(acc_with_loop - acc_without_loop):.4f}")
        
    elif args.ablation == 'normalization':
        print("=== Ablation Study: Normalization Effect ===")
        
        # Load data with symmetric normalization
        adj_symmetric, features, labels, idx_train, idx_val, idx_test = load_data_with_config(
            add_self_loop=True, normalization="symmetric"
        )
        
        # Load data with left normalization
        adj_left, _, _, _, _, _ = load_data_with_config(
            add_self_loop=True, normalization="left"
        )
        
        # Train with symmetric normalization
        print("\n1. Training with symmetric normalization:")
        history_symmetric, acc_symmetric = train_model(adj_symmetric, features, labels, idx_train, idx_val, idx_test, args)
        
        # Train with left normalization
        print("\n2. Training with left normalization:")
        history_left, acc_left = train_model(adj_left, features, labels, idx_train, idx_val, idx_test, args)
        
        # Plot results
        plot_training_curves(
            [history_symmetric, history_left],
            ['Symmetric Normalization', 'Left Normalization'],
            'Normalization Ablation Study',
            './results/normalization_ablation.png'
        )
        
        # Save results
        results = pd.DataFrame({
            'Configuration': ['Symmetric Normalization', 'Left Normalization'],
            'Test Accuracy': [acc_symmetric, acc_left]
        })
        results.to_csv('./results/normalization_results.csv', index=False)
        
        print("\n=== Normalization Ablation Results ===")
        print(results)
        print(f"\nAccuracy difference: {abs(acc_symmetric - acc_left):.4f}")

if __name__ == "__main__":
    main()