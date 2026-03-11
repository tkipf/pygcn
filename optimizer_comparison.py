from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path='./data/cora/')

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Define optimizers to compare
optimizers_config = {
    'Adam': lambda params: optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay),
    'SGD': lambda params: optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay),
    'RMSprop': lambda params: optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay),
    'AdamW': lambda params: optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay),
    'Adagrad': lambda params: optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
}

def train_epoch(model, optimizer, features, adj, labels, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    return {
        'loss_train': loss_train.item(),
        'acc_train': acc_train.item(),
        'loss_val': loss_val.item(),
        'acc_val': acc_val.item(),
        'time': time.time() - t
    }

def test_model(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()

def run_experiment(optimizer_name, optimizer_fn):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    
    if args.cuda:
        model.cuda()
    
    optimizer = optimizer_fn(model.parameters())
    
    history = {
        'loss_train': [],
        'acc_train': [],
        'loss_val': [],
        'acc_val': [],
        'epoch_time': []
    }
    
    print(f"\nTraining with {optimizer_name}...")
    t_total = time.time()
    
    for epoch in range(args.epochs):
        metrics = train_epoch(model, optimizer, features, adj, labels, idx_train, idx_val)
        history['loss_train'].append(metrics['loss_train'])
        history['acc_train'].append(metrics['acc_train'])
        history['loss_val'].append(metrics['loss_val'])
        history['acc_val'].append(metrics['acc_val'])
        history['epoch_time'].append(metrics['time'])
        
        if (epoch + 1) % 20 == 0:
            print(f'{optimizer_name} - Epoch: {epoch+1:04d}, loss_train: {metrics["loss_train"]:.4f}, '
                  f'acc_train: {metrics["acc_train"]:.4f}, loss_val: {metrics["loss_val"]:.4f}, '
                  f'acc_val: {metrics["acc_val"]:.4f}')
    
    total_time = time.time() - t_total
    loss_test, acc_test = test_model(model, features, adj, labels, idx_test)
    
    print(f"{optimizer_name} - Optimization Finished!")
    print(f"Total time elapsed: {total_time:.4f}s")
    print(f"Test set results: loss= {loss_test:.4f}, accuracy= {acc_test:.4f}")
    
    return {
        'history': history,
        'total_time': total_time,
        'loss_test': loss_test,
        'acc_test': acc_test,
        'avg_epoch_time': np.mean(history['epoch_time'])
    }

def plot_matplotlib(results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Comparison - GCN Training', fontsize=16, y=0.95)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Training Loss
    ax = axes[0, 0]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(result['history']['loss_train'], label=opt_name, color=colors[i], alpha=0.8)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(result['history']['loss_val'], label=opt_name, color=colors[i], alpha=0.8)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(result['history']['acc_train'], label=opt_name, color=colors[i], alpha=0.8)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 1]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(result['history']['acc_val'], label=opt_name, color=colors[i], alpha=0.8)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_matplotlib.png', dpi=300, bbox_inches='tight')
    print("Matplotlib figure saved as 'optimizer_comparison_matplotlib.png'")

def plot_plotly(results):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss', 
                       'Training Accuracy', 'Validation Accuracy'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (opt_name, result) in enumerate(results.items()):
        epochs = list(range(1, args.epochs + 1))
        
        # Training Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['loss_train'], 
                      name=opt_name, legendgroup=opt_name,
                      line=dict(color=colors[i], width=2),
                      mode='lines'),
            row=1, col=1
        )
        
        # Validation Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['loss_val'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i], width=2),
                      mode='lines'),
            row=1, col=2
        )
        
        # Training Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['acc_train'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i], width=2),
                      mode='lines'),
            row=2, col=1
        )
        
        # Validation Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['acc_val'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i], width=2),
                      mode='lines'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Optimizer Comparison - GCN Training',
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)
    
    fig.write_html('optimizer_comparison_plotly.html')
    print("Plotly interactive figure saved as 'optimizer_comparison_plotly.html'")

def plot_summary(results):
    summary_data = []
    for opt_name, result in results.items():
        summary_data.append({
            'Optimizer': opt_name,
            'Total_Time': result['total_time'],
            'Avg_Epoch_Time': result['avg_epoch_time'],
            'Test_Loss': result['loss_test'],
            'Test_Accuracy': result['acc_test'],
            'Final_Train_Loss': result['history']['loss_train'][-1],
            'Final_Val_Loss': result['history']['loss_val'][-1],
            'Final_Train_Acc': result['history']['acc_train'][-1],
            'Final_Val_Acc': result['history']['acc_val'][-1]
        })
    
    df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Performance Summary', fontsize=16, y=0.95)
    
    # Test Accuracy
    ax = axes[0, 0]
    sns.barplot(x='Optimizer', y='Test_Accuracy', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.set_title('Test Accuracy')
    ax.set_ylim(df['Test_Accuracy'].min() - 0.05, df['Test_Accuracy'].max() + 0.02)
    ax.tick_params(axis='x', rotation=45)
    
    # Test Loss
    ax = axes[0, 1]
    sns.barplot(x='Optimizer', y='Test_Loss', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.set_title('Test Loss')
    ax.tick_params(axis='x', rotation=45)
    
    # Total Training Time
    ax = axes[1, 0]
    sns.barplot(x='Optimizer', y='Total_Time', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.set_title('Total Training Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    
    # Average Epoch Time
    ax = axes[1, 1]
    sns.barplot(x='Optimizer', y='Avg_Epoch_Time', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.set_title('Average Epoch Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('optimizer_summary.png', dpi=300, bbox_inches='tight')
    print("Summary figure saved as 'optimizer_summary.png'")
    
    return df

def print_summary_table(df):
    print("\n" + "="*80)
    print("OPTIMIZER PERFORMANCE SUMMARY")
    print("="*80)
    
    cols = ['Optimizer', 'Test_Accuracy', 'Test_Loss', 'Total_Time', 'Avg_Epoch_Time']
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Rank optimizers
    print("\n" + "="*80)
    print("OPTIMIZER RANKING")
    print("="*80)
    
    # Composite score: higher accuracy, lower loss, lower time
    df['rank_acc'] = df['Test_Accuracy'].rank(ascending=False)
    df['rank_loss'] = df['Test_Loss'].rank(ascending=True)
    df['rank_time'] = df['Total_Time'].rank(ascending=True)
    df['composite_score'] = (df['rank_acc'] + df['rank_loss'] + df['rank_time']) / 3
    
    df_ranked = df.sort_values('composite_score')
    print(df_ranked[['Optimizer', 'Test_Accuracy', 'Test_Loss', 'Total_Time', 'composite_score']]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    best_opt = df_ranked.iloc[0]['Optimizer']
    print(f"\nBest Optimizer: {best_opt}")
    print(f"  - Test Accuracy: {df_ranked.iloc[0]['Test_Accuracy']:.4f}")
    print(f"  - Test Loss: {df_ranked.iloc[0]['Test_Loss']:.4f}")
    print(f"  - Total Training Time: {df_ranked.iloc[0]['Total_Time']:.4f}s")
    
    return df_ranked

def main():
    results = {}
    
    for opt_name, opt_fn in optimizers_config.items():
        results[opt_name] = run_experiment(opt_name, opt_fn)
    
    # Visualization
    plot_matplotlib(results)
    plot_plotly(results)
    df = plot_summary(results)
    
    # Summary
    df_ranked = print_summary_table(df)
    
    # Save results to CSV
    df.to_csv('optimizer_results.csv', index=False)
    print("\nResults saved to 'optimizer_results.csv'")
    
    return results, df_ranked

if __name__ == "__main__":
    main()
