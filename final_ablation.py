from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data using the original load_data function to ensure consistency
# This gives us the exact same setup as the original train.py
adj, features, labels, idx_train, idx_val, idx_test = load_data(path="./data/cora/")

# Create a copy of the original adjacency matrix for reference
original_adj = adj

# Function to create adjacency matrix without self-loops
def create_adj_no_selfloop():
    # Load raw data again to create adjacency matrix without self-loops
    path = "./data/cora/"
    dataset = "cora"
    
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    
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
    
    # Normalize without adding self-loop
    adj = normalize(adj)
    
    # Convert to torch sparse tensor
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj_tensor

# Function to create symmetric normalized adjacency matrix
def create_adj_symmetric_norm():
    # Load raw data again to create symmetric normalized adjacency matrix
    path = "./data/cora/"
    dataset = "cora"
    
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    
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
    
    # Symmetric normalization with self-loop
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    adj = adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    
    # Convert to torch sparse tensor
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj_tensor

# Create adjacency matrices for ablation
adj_no_selfloop = create_adj_no_selfloop()
adj_symmetric_norm = create_adj_symmetric_norm()

def train_model(adj_tensor, model_name):
    """Train GCN model with given adjacency matrix"""
    if args.cuda:
        adj_tensor = adj_tensor.cuda()
    
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    
    if args.cuda:
        model.cuda()
    
    # Training history
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    # Train with tqdm
    print(f"\nTraining {model_name}...")
    for epoch in tqdm(range(args.epochs), desc=model_name):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_tensor)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        if not args.fastmode:
            # Evaluate validation set performance separately
            model.eval()
            output = model(features, adj_tensor)
        
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        # Save history
        train_loss_history.append(loss_train.item())
        train_acc_history.append(acc_train.item())
        val_loss_history.append(loss_val.item())
        val_acc_history.append(acc_val.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:04d}',
                  f'loss_train: {loss_train.item():.4f}',
                  f'acc_train: {acc_train.item():.4f}',
                  f'loss_val: {loss_val.item():.4f}',
                  f'acc_val: {acc_val.item():.4f}')
    
    # Test
    model.eval()
    output = model(features, adj_tensor)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    print(f"\n{model_name} Test Results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
    return {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'test_acc': acc_test.item()
    }

# Run experiments
print("\n=== Ablation Study ===")

# Baseline: Original model with self-loop and left normalization
print("\n1. Baseline: Left normalization with self-loop")
result_baseline = train_model(original_adj, "baseline")

# Without self-loop
print("\n2. Without self-loop (left normalization)")
result_no_selfloop = train_model(adj_no_selfloop, "no_selfloop")

# Symmetric normalization with self-loop
print("\n3. Symmetric normalization with self-loop")
result_symmetric = train_model(adj_symmetric_norm, "symmetric_norm")

# Save results to CSV
results_df = pd.DataFrame({
    'Model': ['Baseline (Left Norm + Self-loop)', 'No Self-loop', 'Symmetric Norm + Self-loop'],
    'Test Accuracy': [
        result_baseline['test_acc'],
        result_no_selfloop['test_acc'],
        result_symmetric['test_acc']
    ]
})

results_df.to_csv('final_ablation_results.csv', index=False)
print("\nResults saved to final_ablation_results.csv")

# Print summary
print("\n=== Final Ablation Study Summary ===")
print(results_df.to_string(index=False))

print("\nKey Findings:")
print(f"1. Self-loop effect: {'improves' if result_baseline['test_acc'] > result_no_selfloop['test_acc'] else 'degrades'} accuracy by {abs(result_baseline['test_acc'] - result_no_selfloop['test_acc'])*100:.2f} percentage points")
print(f"2. Symmetric normalization achieves {result_symmetric['test_acc']*100:.2f}% test accuracy")
print(f"3. Left normalization achieves {result_baseline['test_acc']*100:.2f}% test accuracy")
print(f"4. Best performing model: {'Symmetric Norm' if result_symmetric['test_acc'] > result_baseline['test_acc'] else 'Left Norm (Baseline)'}")