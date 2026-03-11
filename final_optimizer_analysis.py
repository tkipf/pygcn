import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('optimizer_results_v2.csv')
conv_df = pd.read_csv('optimizer_convergence.csv')

# Extract needed columns from convergence data
conv_df = conv_df[['Optimizer', 'Epochs_to_80pct']]

# Merge dataframes
df = df.merge(conv_df, on='Optimizer')

# Filter out poor performing optimizers
good_optimizers = df[df['Test_Accuracy'] > 0.7].copy()

print("="*100)
print("DETAILED ANALYSIS OF GOOD PERFORMING OPTIMIZERS")
print("="*100)

# Normalize metrics for scoring
def normalize_col(col, higher_is_better=True):
    if higher_is_better:
        return (col - col.min()) / (col.max() - col.min())
    else:
        return (col.max() - col) / (col.max() - col.min())

# Calculate comprehensive scores
# 1. Performance (40% weight)
#    - Test Accuracy (25%)
#    - Test Loss (15%)
# 2. Speed (25% weight)
#    - Total Time (15%)
#    - Avg Epoch Time (10%)
# 3. Stability (20% weight)
#    - Accuracy Std (10%)
#    - Loss Std (5%)
#    - Time Std (5%)
# 4. Convergence (15% weight)
#    - Epochs to 80% accuracy (10%)
#    - Final Validation Accuracy (5%)

good_optimizers.loc[:, 'score_acc'] = normalize_col(good_optimizers['Test_Accuracy'], True)
good_optimizers.loc[:, 'score_loss'] = normalize_col(good_optimizers['Test_Loss'], False)
good_optimizers.loc[:, 'score_time'] = normalize_col(good_optimizers['Total_Time'], False)
good_optimizers.loc[:, 'score_epoch_time'] = normalize_col(good_optimizers['Avg_Epoch_Time'], False)
good_optimizers.loc[:, 'score_acc_std'] = normalize_col(good_optimizers['Test_Accuracy_Std'], False)
good_optimizers.loc[:, 'score_loss_std'] = normalize_col(good_optimizers['Test_Loss_Std'], False)
good_optimizers.loc[:, 'score_time_std'] = normalize_col(good_optimizers['Total_Time_Std'], False)
good_optimizers.loc[:, 'score_convergence'] = normalize_col(good_optimizers['Epochs_to_80pct'], False)
good_optimizers.loc[:, 'score_final_val_acc'] = normalize_col(good_optimizers['Final_Val_Acc'], True)

# Weighted composite score
good_optimizers.loc[:, 'composite_score'] = (
    good_optimizers['score_acc'] * 0.25 +
    good_optimizers['score_loss'] * 0.15 +
    good_optimizers['score_time'] * 0.15 +
    good_optimizers['score_epoch_time'] * 0.10 +
    good_optimizers['score_acc_std'] * 0.10 +
    good_optimizers['score_loss_std'] * 0.05 +
    good_optimizers['score_time_std'] * 0.05 +
    good_optimizers['score_convergence'] * 0.10 +
    good_optimizers['score_final_val_acc'] * 0.05
)

# Sort by composite score
good_optimizers = good_optimizers.sort_values('composite_score', ascending=False).reset_index(drop=True)

print("\nDetailed Metrics:")
print("-" * 100)
cols = ['Optimizer', 'Test_Accuracy', 'Test_Accuracy_Std', 'Test_Loss', 'Test_Loss_Std', 
        'Total_Time', 'Total_Time_Std', 'Avg_Epoch_Time', 'Epochs_to_80pct', 'Final_Val_Acc']
print(good_optimizers[cols].to_string(index=False))

print("\n" + "="*100)
print("COMPREHENSIVE SCORING (higher is better)")
print("="*100)

score_cols = ['Optimizer', 'score_acc', 'score_loss', 'score_time', 'score_epoch_time',
              'score_acc_std', 'score_loss_std', 'score_time_std', 'score_convergence', 
              'score_final_val_acc', 'composite_score']
print(good_optimizers[score_cols].to_string(index=False))

# Visualize radar chart
plt.figure(figsize=(12, 10))
categories = ['Accuracy', 'Loss', 'Speed', 'Epoch Speed', 'Acc Stability', 
              'Loss Stability', 'Time Stability', 'Convergence', 'Val Accuracy']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (idx, row) in enumerate(good_optimizers.iterrows()):
    values = [
        row['score_acc'], row['score_loss'], row['score_time'], row['score_epoch_time'],
        row['score_acc_std'], row['score_loss_std'], row['score_time_std'],
        row['score_convergence'], row['score_final_val_acc']
    ]
    values += values[:1]
    ax.plot(angles, values, label=row['Optimizer'], color=colors[i], linewidth=2, alpha=0.8)
    ax.fill(angles, values, color=colors[i], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Optimizer Performance Radar Chart (Normalized Scores)', size=16, y=1.1)
ax.legend(bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('optimizer_radar.png', dpi=300, bbox_inches='tight')
print("\nRadar chart saved as 'optimizer_radar.png'")

# Create bar chart for final ranking
plt.figure(figsize=(12, 6))
bars = sns.barplot(x='Optimizer', y='composite_score', hue='Optimizer', data=good_optimizers, palette='viridis', legend=False)
plt.title('Optimizer Composite Scores (Weighted)', fontsize=16)
plt.ylabel('Composite Score (higher = better)')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)

# Add value labels
for i, v in enumerate(good_optimizers['composite_score']):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('optimizer_final_ranking.png', dpi=300, bbox_inches='tight')
print("Final ranking chart saved as 'optimizer_final_ranking.png'")

# Print final recommendation
print("\n" + "="*100)
print("FINAL RECOMMENDATION")
print("="*100)

best = good_optimizers.iloc[0]
print(f"\n🏆 BEST OPTIMIZER: {best['Optimizer']}")
print("-" * 50)
print(f"  Test Accuracy:    {best['Test_Accuracy']:.4f} ± {best['Test_Accuracy_Std']:.4f}")
print(f"  Test Loss:        {best['Test_Loss']:.4f} ± {best['Test_Loss_Std']:.4f}")
print(f"  Total Time:       {best['Total_Time']:.4f} ± {best['Total_Time_Std']:.4f}s")
print(f"  Avg Epoch Time:   {best['Avg_Epoch_Time']*1000:.2f}ms")
print(f"  Epochs to 80%:    {best['Epochs_to_80pct']}")
print(f"  Composite Score:  {best['composite_score']:.4f}")

print("\n📊 Rankings Summary:")
for i, (idx, row) in enumerate(good_optimizers.iterrows()):
    print(f"  {i+1}. {row['Optimizer']}: {row['composite_score']:.4f} (Acc: {row['Test_Accuracy']:.4f}, Time: {row['Total_Time']:.4f}s)")

print("\n" + "="*100)
print("ANALYSIS BY CATEGORY")
print("="*100)

print("\n🎯 Best Accuracy:")
acc_best = good_optimizers.loc[good_optimizers['Test_Accuracy'].idxmax()]
print(f"  {acc_best['Optimizer']}: {acc_best['Test_Accuracy']:.4f} ± {acc_best['Test_Accuracy_Std']:.4f}")

print("\n⚡ Best Speed (Fastest Training):")
speed_best = good_optimizers.loc[good_optimizers['Total_Time'].idxmin()]
print(f"  {speed_best['Optimizer']}: {speed_best['Total_Time']:.4f} ± {speed_best['Total_Time_Std']:.4f}s")

print("\n📉 Best Loss:")
loss_best = good_optimizers.loc[good_optimizers['Test_Loss'].idxmin()]
print(f"  {loss_best['Optimizer']}: {loss_best['Test_Loss']:.4f} ± {loss_best['Test_Loss_Std']:.4f}")

print("\n🔒 Most Stable (Lowest Accuracy Std):")
stable_best = good_optimizers.loc[good_optimizers['Test_Accuracy_Std'].idxmin()]
print(f"  {stable_best['Optimizer']}: {stable_best['Test_Accuracy_Std']:.6f}")

print("\n🚀 Fastest Convergence (Epochs to 80%):")
conv_best = good_optimizers.loc[good_optimizers['Epochs_to_80pct'].idxmin()]
print(f"  {conv_best['Optimizer']}: {conv_best['Epochs_to_80pct']} epochs")

print("\n💡 Recommendation Notes:")
print("  - Adam excels in accuracy and stability")
print("  - Adagrad offers good accuracy with faster training")  
print("  - RMSprop converges the fastest but has higher time variance")
print("  - AdamW has the lowest loss but slightly lower accuracy")
print("  - SGD variants perform poorly for this GCN task")

# Save detailed analysis
good_optimizers.to_csv('optimizer_detailed_analysis.csv', index=False)
print("\nDetailed analysis saved to 'optimizer_detailed_analysis.csv'")
