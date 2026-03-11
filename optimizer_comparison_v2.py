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
parser.add_argument('--runs', type=int, default=3,
                    help='Number of runs for each optimizer to average results.')

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

# Define optimizers with tuned hyperparameters
optimizers_config = {
    'Adam': {
        'optimizer': lambda params, lr, wd: optim.Adam(params, lr=lr, weight_decay=wd),
        'lr': 0.01,
        'weight_decay': 5e-4
    },
    'Adam (lr=0.001)': {
        'optimizer': lambda params, lr, wd: optim.Adam(params, lr=lr, weight_decay=wd),
        'lr': 0.001,
        'weight_decay': 5e-4
    },
    'SGD (momentum=0.9)': {
        'optimizer': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd),
        'lr': 0.1,
        'weight_decay': 5e-4
    },
    'SGD (lr=0.5)': {
        'optimizer': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd),
        'lr': 0.5,
        'weight_decay': 5e-4
    },
    'RMSprop': {
        'optimizer': lambda params, lr, wd: optim.RMSprop(params, lr=lr, weight_decay=wd),
        'lr': 0.01,
        'weight_decay': 5e-4
    },
    'AdamW': {
        'optimizer': lambda params, lr, wd: optim.AdamW(params, lr=lr, weight_decay=wd),
        'lr': 0.01,
        'weight_decay': 5e-4
    },
    'Adagrad': {
        'optimizer': lambda params, lr, wd: optim.Adagrad(params, lr=lr, weight_decay=wd),
        'lr': 0.1,
        'weight_decay': 5e-4
    }
}

def create_model():
    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    if args.cuda:
        model.cuda()
    return model

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

def run_single_experiment(optimizer_name, opt_config, run_seed):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if args.cuda:
        torch.cuda.manual_seed(run_seed)
    
    model = create_model()
    optimizer = opt_config['optimizer'](model.parameters(), opt_config['lr'], opt_config['weight_decay'])
    
    history = {
        'loss_train': [],
        'acc_train': [],
        'loss_val': [],
        'acc_val': [],
        'epoch_time': []
    }
    
    t_total = time.time()
    
    for epoch in range(args.epochs):
        metrics = train_epoch(model, optimizer, features, adj, labels, idx_train, idx_val)
        history['loss_train'].append(metrics['loss_train'])
        history['acc_train'].append(metrics['acc_train'])
        history['loss_val'].append(metrics['loss_val'])
        history['acc_val'].append(metrics['acc_val'])
        history['epoch_time'].append(metrics['time'])
    
    total_time = time.time() - t_total
    loss_test, acc_test = test_model(model, features, adj, labels, idx_test)
    
    return {
        'history': history,
        'total_time': total_time,
        'loss_test': loss_test,
        'acc_test': acc_test,
        'avg_epoch_time': np.mean(history['epoch_time'])
    }

def run_experiment(optimizer_name, opt_config):
    all_results = []
    all_histories = []
    
    print(f"\nTraining with {optimizer_name}...")
    
    for run in range(args.runs):
        run_seed = args.seed + run * 100
        result = run_single_experiment(optimizer_name, opt_config, run_seed)
        all_results.append(result)
        all_histories.append(result['history'])
        print(f"  Run {run+1}/{args.runs}: Test Acc = {result['acc_test']:.4f}, Time = {result['total_time']:.4f}s")
    
    # Average histories
    avg_history = {
        'loss_train': np.mean([h['loss_train'] for h in all_histories], axis=0),
        'acc_train': np.mean([h['acc_train'] for h in all_histories], axis=0),
        'loss_val': np.mean([h['loss_val'] for h in all_histories], axis=0),
        'acc_val': np.mean([h['acc_val'] for h in all_histories], axis=0),
        'loss_train_std': np.std([h['loss_train'] for h in all_histories], axis=0),
        'acc_train_std': np.std([h['acc_train'] for h in all_histories], axis=0),
        'loss_val_std': np.std([h['loss_val'] for h in all_histories], axis=0),
        'acc_val_std': np.std([h['acc_val'] for h in all_histories], axis=0)
    }
    
    # Average metrics
    avg_result = {
        'history': avg_history,
        'total_time': np.mean([r['total_time'] for r in all_results]),
        'total_time_std': np.std([r['total_time'] for r in all_results]),
        'loss_test': np.mean([r['loss_test'] for r in all_results]),
        'loss_test_std': np.std([r['loss_test'] for r in all_results]),
        'acc_test': np.mean([r['acc_test'] for r in all_results]),
        'acc_test_std': np.std([r['acc_test'] for r in all_results]),
        'avg_epoch_time': np.mean([r['avg_epoch_time'] for r in all_results]),
        'all_runs': all_results
    }
    
    print(f"{optimizer_name} - Average Results:")
    print(f"  Test Accuracy: {avg_result['acc_test']:.4f} ± {avg_result['acc_test_std']:.4f}")
    print(f"  Test Loss: {avg_result['loss_test']:.4f} ± {avg_result['loss_test_std']:.4f}")
    print(f"  Total Time: {avg_result['total_time']:.4f} ± {avg_result['total_time_std']:.4f}s")
    
    return avg_result

def plot_matplotlib(results, save_dir='.'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Comparison - GCN Training (Average over 3 runs)', fontsize=16, y=0.95)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    epochs = np.arange(1, args.epochs + 1)
    
    # Training Loss
    ax = axes[0, 0]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(epochs, result['history']['loss_train'], label=opt_name, color=colors[i], alpha=0.8)
        ax.fill_between(epochs, 
                       result['history']['loss_train'] - result['history']['loss_train_std'],
                       result['history']['loss_train'] + result['history']['loss_train_std'],
                       color=colors[i], alpha=0.2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(epochs, result['history']['loss_val'], label=opt_name, color=colors[i], alpha=0.8)
        ax.fill_between(epochs, 
                       result['history']['loss_val'] - result['history']['loss_val_std'],
                       result['history']['loss_val'] + result['history']['loss_val_std'],
                       color=colors[i], alpha=0.2)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(epochs, result['history']['acc_train'], label=opt_name, color=colors[i], alpha=0.8)
        ax.fill_between(epochs, 
                       result['history']['acc_train'] - result['history']['acc_train_std'],
                       result['history']['acc_train'] + result['history']['acc_train_std'],
                       color=colors[i], alpha=0.2)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 1]
    for i, (opt_name, result) in enumerate(results.items()):
        ax.plot(epochs, result['history']['acc_val'], label=opt_name, color=colors[i], alpha=0.8)
        ax.fill_between(epochs, 
                       result['history']['acc_val'] - result['history']['acc_val_std'],
                       result['history']['acc_val'] + result['history']['acc_val_std'],
                       color=colors[i], alpha=0.2)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimizer_comparison_v2_matplotlib.png', dpi=300, bbox_inches='tight')
    print(f"Matplotlib figure saved as '{save_dir}/optimizer_comparison_v2_matplotlib.png'")

def plot_plotly(results, save_dir='.'):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss', 
                       'Training Accuracy', 'Validation Accuracy'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    epochs = list(range(1, args.epochs + 1))
    
    for i, (opt_name, result) in enumerate(results.items()):
        # Training Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['loss_train'], 
                      name=opt_name, legendgroup=opt_name,
                      line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines'),
            row=1, col=1
        )
        
        # Validation Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['loss_val'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines'),
            row=1, col=2
        )
        
        # Training Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['acc_train'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines'),
            row=2, col=1
        )
        
        # Validation Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=result['history']['acc_val'], 
                      name=opt_name, legendgroup=opt_name, showlegend=False,
                      line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Optimizer Comparison - GCN Training (Average over 3 runs)',
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
    
    fig.write_html(f'{save_dir}/optimizer_comparison_v2_plotly.html')
    print(f"Plotly interactive figure saved as '{save_dir}/optimizer_comparison_v2_plotly.html'")

def plot_summary(results, save_dir='.'):
    summary_data = []
    for opt_name, result in results.items():
        summary_data.append({
            'Optimizer': opt_name,
            'Total_Time': result['total_time'],
            'Total_Time_Std': result['total_time_std'],
            'Avg_Epoch_Time': result['avg_epoch_time'],
            'Test_Loss': result['loss_test'],
            'Test_Loss_Std': result['loss_test_std'],
            'Test_Accuracy': result['acc_test'],
            'Test_Accuracy_Std': result['acc_test_std'],
            'Final_Train_Loss': result['history']['loss_train'][-1],
            'Final_Val_Loss': result['history']['loss_val'][-1],
            'Final_Train_Acc': result['history']['acc_train'][-1],
            'Final_Val_Acc': result['history']['acc_val'][-1]
        })
    
    df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Performance Summary (Average over 3 runs)', fontsize=16, y=0.95)
    
    # Test Accuracy with error bars
    ax = axes[0, 0]
    bars = sns.barplot(x='Optimizer', y='Test_Accuracy', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.errorbar(x=np.arange(len(df)), y=df['Test_Accuracy'], yerr=df['Test_Accuracy_Std'], 
                fmt='none', c='black', capsize=5)
    ax.set_title('Test Accuracy')
    ax.set_ylim(max(0, df['Test_Accuracy'].min() - 0.1), min(1, df['Test_Accuracy'].max() + 0.05))
    ax.tick_params(axis='x', rotation=45)
    
    # Test Loss with error bars
    ax = axes[0, 1]
    sns.barplot(x='Optimizer', y='Test_Loss', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.errorbar(x=np.arange(len(df)), y=df['Test_Loss'], yerr=df['Test_Loss_Std'], 
                fmt='none', c='black', capsize=5)
    ax.set_title('Test Loss')
    ax.tick_params(axis='x', rotation=45)
    
    # Total Training Time with error bars
    ax = axes[1, 0]
    sns.barplot(x='Optimizer', y='Total_Time', hue='Optimizer', data=df, ax=ax, palette='viridis', legend=False)
    ax.errorbar(x=np.arange(len(df)), y=df['Total_Time'], yerr=df['Total_Time_Std'], 
                fmt='none', c='black', capsize=5)
    ax.set_title('Total Training Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    
    # Speed vs Accuracy scatter plot
    ax = axes[1, 1]
    for i, row in df.iterrows():
        ax.scatter(row['Total_Time'], row['Test_Accuracy'], s=200, label=row['Optimizer'], alpha=0.7)
        ax.errorbar(row['Total_Time'], row['Test_Accuracy'], 
                   xerr=row['Total_Time_Std'], yerr=row['Test_Accuracy_Std'],
                   fmt='none', alpha=0.5)
    ax.set_title('Speed vs Accuracy Trade-off')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimizer_summary_v2.png', dpi=300, bbox_inches='tight')
    print(f"Summary figure saved as '{save_dir}/optimizer_summary_v2.png'")
    
    return df

def print_summary_table(df):
    print("\n" + "="*120)
    print("OPTIMIZER PERFORMANCE SUMMARY (AVERAGE OVER 3 RUNS)")
    print("="*120)
    
    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    cols = ['Optimizer', 'Test_Accuracy', 'Test_Accuracy_Std', 'Test_Loss', 'Test_Loss_Std', 'Total_Time', 'Total_Time_Std']
    print(df[cols].to_string(index=False))
    
    # Rank optimizers - lower score is better
    print("\n" + "="*120)
    print("OPTIMIZER RANKING")
    print("="*120)
    
    # Composite score: higher accuracy, lower loss, lower time
    df['rank_acc'] = df['Test_Accuracy'].rank(ascending=False)
    df['rank_loss'] = df['Test_Loss'].rank(ascending=True)
    df['rank_time'] = df['Total_Time'].rank(ascending=True)
    df['composite_score'] = (df['rank_acc'] + df['rank_loss'] + df['rank_time']) / 3
    
    # Also consider stability (lower std is better)
    df['rank_acc_std'] = df['Test_Accuracy_Std'].rank(ascending=True)
    df['composite_score_with_stability'] = (df['rank_acc'] + df['rank_loss'] + df['rank_time'] + df['rank_acc_std']) / 4
    
    df_ranked = df.sort_values('composite_score')
    print("\nRanking based on (Accuracy + Loss + Time):")
    cols_rank = ['Optimizer', 'Test_Accuracy', 'Test_Loss', 'Total_Time', 'composite_score']
    print(df_ranked[cols_rank].to_string(index=False))
    
    df_ranked_stability = df.sort_values('composite_score_with_stability')
    print("\nRanking based on (Accuracy + Loss + Time + Stability):")
    cols_rank_stab = ['Optimizer', 'Test_Accuracy', 'Test_Accuracy_Std', 'Test_Loss', 'Total_Time', 'composite_score_with_stability']
    print(df_ranked_stability[cols_rank_stab].to_string(index=False))
    
    best_opt = df_ranked.iloc[0]['Optimizer']
    best_opt_stab = df_ranked_stability.iloc[0]['Optimizer']
    
    print(f"\nBest Optimizer (without stability): {best_opt}")
    print(f"  - Test Accuracy: {df_ranked.iloc[0]['Test_Accuracy']:.4f} ± {df_ranked.iloc[0]['Test_Accuracy_Std']:.4f}")
    print(f"  - Test Loss: {df_ranked.iloc[0]['Test_Loss']:.4f} ± {df_ranked.iloc[0]['Test_Loss_Std']:.4f}")
    print(f"  - Total Training Time: {df_ranked.iloc[0]['Total_Time']:.4f} ± {df_ranked.iloc[0]['Total_Time_Std']:.4f}s")
    
    print(f"\nBest Optimizer (with stability): {best_opt_stab}")
    print(f"  - Test Accuracy: {df_ranked_stability.iloc[0]['Test_Accuracy']:.4f} ± {df_ranked_stability.iloc[0]['Test_Accuracy_Std']:.4f}")
    print(f"  - Test Loss: {df_ranked_stability.iloc[0]['Test_Loss']:.4f} ± {df_ranked_stability.iloc[0]['Test_Loss_Std']:.4f}")
    print(f"  - Total Training Time: {df_ranked_stability.iloc[0]['Total_Time']:.4f} ± {df_ranked_stability.iloc[0]['Total_Time_Std']:.4f}s")
    
    return df_ranked, df_ranked_stability

def plot_convergence_analysis(results, save_dir='.'):
    """Analyze how fast each optimizer converges"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = np.arange(1, args.epochs + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Epochs to reach 80% of best accuracy
    target_acc_ratio = 0.8
    
    ax1.set_title('Validation Accuracy Convergence')
    ax2.set_title('Training Loss Convergence (Log Scale)')
    
    convergence_data = []
    
    for i, (opt_name, result) in enumerate(results.items()):
        ax1.plot(epochs, result['history']['acc_val'], label=opt_name, color=colors[i], alpha=0.8)
        
        # Find epochs to reach 80% of final accuracy
        final_acc = result['history']['acc_val'][-1]
        target_acc = 0.8 * final_acc
        epochs_to_target = np.argmax(result['history']['acc_val'] >= target_acc)
        if epochs_to_target == 0 and result['history']['acc_val'][0] < target_acc:
            epochs_to_target = args.epochs
        
        # Loss convergence (log scale)
        ax2.plot(epochs, np.log(result['history']['loss_train']), label=opt_name, color=colors[i], alpha=0.8)
        
        convergence_data.append({
            'Optimizer': opt_name,
            'Final_Val_Acc': final_acc,
            'Epochs_to_80pct': epochs_to_target,
            'Final_Train_Loss': result['history']['loss_train'][-1]
        })
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Log(Training Loss)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimizer_convergence.png', dpi=300, bbox_inches='tight')
    print(f"Convergence analysis figure saved as '{save_dir}/optimizer_convergence.png'")
    
    return pd.DataFrame(convergence_data)

def main():
    results = {}
    
    for opt_name, opt_config in optimizers_config.items():
        results[opt_name] = run_experiment(opt_name, opt_config)
    
    # Visualization
    plot_matplotlib(results)
    plot_plotly(results)
    df = plot_summary(results)
    conv_df = plot_convergence_analysis(results)
    
    # Summary
    df_ranked, df_ranked_stab = print_summary_table(df)
    
    print("\nConvergence Analysis:")
    print(conv_df.to_string(index=False))
    
    # Save results to CSV
    df.to_csv('optimizer_results_v2.csv', index=False)
    conv_df.to_csv('optimizer_convergence.csv', index=False)
    print("\nResults saved to 'optimizer_results_v2.csv' and 'optimizer_convergence.csv'")
    
    return results, df_ranked, df_ranked_stab

if __name__ == "__main__":
    main()
