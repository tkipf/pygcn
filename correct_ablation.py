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
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# Load raw data without normalization
path = "./data/cora/"
dataset = "cora"

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

# Normalize features
features = normalize(features)

# Fixed splits
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)

features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

if args.cuda:
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def symmetric_normalize(mx):
    """Symmetric normalize adjacency matrix"""
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def left_normalize(mx):
    """Left normalize adjacency matrix"""
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
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
        
        # Evaluate validation set performance
        model.eval()
        output = model(features, adj_tensor)
        
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        # Save history
        train_loss_history.append(loss_train.item())
        train_acc_history.append(acc_train.item())
        val_loss_history.append(loss_val.item())
        val_acc_history.append(acc_val.item())
        
        # Early stopping
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"best_model_{model_name}.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and test
    model.load_state_dict(torch.load(f"best_model_{model_name}.pth"))
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
        'test_acc': acc_test.item(),
        'best_epoch': best_epoch
    }


# Create different adjacency matrices for ablation
print("\n=== Preparing adjacency matrices for ablation ===")

# 1. Baseline: Left normalization with self-loop (same as original code)
adj_baseline = normalize(adj + sp.eye(adj.shape[0]))
adj_baseline_tensor = sparse_mx_to_torch_sparse_tensor(adj_baseline)

# 2. Without self-loop (left normalized)
adj_no_selfloop = normalize(adj.copy())
adj_no_selfloop_tensor = sparse_mx_to_torch_sparse_tensor(adj_no_selfloop)

# 3. Symmetric normalization with self-loop
adj_symmetric = symmetric_normalize(adj.copy())
adj_symmetric_tensor = sparse_mx_to_torch_sparse_tensor(adj_symmetric)

# 4. Left normalization with self-loop (same as baseline)
adj_left = left_normalize(adj.copy())
adj_left_tensor = sparse_mx_to_torch_sparse_tensor(adj_left)

# Experiment 1: Self-loop ablation
print("\n=== Experiment 1: Self-loop Ablation ===")

# Baseline: With self-loop (original)
result_baseline = train_model(adj_baseline_tensor, "baseline_with_selfloop")

# Without self-loop
result_no_selfloop = train_model(adj_no_selfloop_tensor, "no_selfloop")

# Experiment 2: Normalization ablation
print("\n=== Experiment 2: Normalization Ablation ===")

# Symmetric normalization (with self-loop)
result_symmetric = train_model(adj_symmetric_tensor, "symmetric_norm")

# Left normalization (with self-loop - same as baseline)
result_left = train_model(adj_left_tensor, "left_norm")

# Save results to CSV
results_df = pd.DataFrame({
    'Model': ['Baseline (Left Norm + Self-loop)', 'No Self-loop', 'Symmetric Norm + Self-loop', 'Left Norm + Self-loop'],
    'Test Accuracy': [
        result_baseline['test_acc'],
        result_no_selfloop['test_acc'],
        result_symmetric['test_acc'],
        result_left['test_acc']
    ],
    'Best Epoch': [
        result_baseline['best_epoch'],
        result_no_selfloop['best_epoch'],
        result_symmetric['best_epoch'],
        result_left['best_epoch']
    ]
})

results_df.to_csv('correct_ablation_results.csv', index=False)
print("\nResults saved to correct_ablation_results.csv")

# Plot training curves
plt.figure(figsize=(12, 8))

# Self-loop comparison
plt.subplot(2, 1, 1)
plt.plot(result_baseline['train_acc'], label='With Self-loop (Train)')
plt.plot(result_baseline['val_acc'], label='With Self-loop (Val)')
plt.plot(result_no_selfloop['train_acc'], label='No Self-loop (Train)')
plt.plot(result_no_selfloop['val_acc'], label='No Self-loop (Val)')
plt.title('Self-loop Ablation: Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Normalization comparison
plt.subplot(2, 1, 2)
plt.plot(result_symmetric['train_acc'], label='Symmetric Norm (Train)')
plt.plot(result_symmetric['val_acc'], label='Symmetric Norm (Val)')
plt.plot(result_left['train_acc'], label='Left Norm (Train)')
plt.plot(result_left['val_acc'], label='Left Norm (Val)')
plt.title('Normalization Ablation: Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('correct_ablation_curves.png', dpi=300)
plt.close()
print("Training curves saved to correct_ablation_curves.png")

# Print summary
print("\n=== Correct Ablation Study Summary ===")
print(results_df.to_string(index=False))

print("\nKey Findings:")
print(f"1. Self-loop improves accuracy by {(result_baseline['test_acc'] - result_no_selfloop['test_acc'])*100:.2f} percentage points")
print(f"2. Symmetric normalization achieves {result_symmetric['test_acc']*100:.2f}% test accuracy")
print(f"3. Left normalization achieves {result_left['test_acc']*100:.2f}% test accuracy")
print(f"4. Best performing model: {'Symmetric Norm' if result_symmetric['test_acc'] > result_left['test_acc'] else 'Left Norm'}")