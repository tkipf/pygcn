# GCN Ablation Study Final Report

## Experimental Setup

### Dataset
- Cora citation network
- 2708 scientific publications
- 5429 links between publications
- 7 classes

### Training Configuration
- 2-layer GCN model
- Hidden units: 16
- Dropout: 0.5
- Optimizer: Adam
- Learning rate: 0.01
- Weight decay: 5e-4
- Training epochs: 200
- Fixed train/val/test splits: 140/300/1000

## Experiment 1: Self-loop Ablation

### Results
| Model | Test Accuracy |
|-------|---------------|
| With Self-loop (Baseline) | 83.5% |
| Without Self-loop | 80.8% |

### Analysis
- **Self-loop improves accuracy by 2.7 percentage points**
- Self-loops allow nodes to retain their own information during propagation
- Without self-loops, nodes only aggregate information from neighbors, leading to slight information loss
- The difference is significant but not dramatic, indicating that the model can still perform reasonably well without self-loops

## Experiment 2: Normalization Ablation

### Results
| Model | Test Accuracy |
|-------|---------------|
| Left Normalization (Baseline) | 83.5% |
| Symmetric Normalization | 81.7% |

### Analysis
- **Left normalization performs better than symmetric normalization on this dataset**
- Left normalization (row normalization) ensures each row sums to 1
- Symmetric normalization (D^-1/2 A D^-1/2) is more commonly used in GCN literature
- The difference of 1.8 percentage points is noticeable but not dramatic

## Key Findings

1. **Self-loops are beneficial but not critical**: They improve accuracy by 2.7 percentage points, but the model still performs well without them
2. **Normalization choice matters**: Left normalization outperforms symmetric normalization on this dataset
3. **The baseline model is the best performer**: Left normalization with self-loop achieves the highest accuracy of 83.5%
4. **All models achieve good performance**: Even the worst performing model (symmetric normalization) achieves 81.7% accuracy

## Visualization

### Training Curves
- All models show similar training trends
- The baseline model converges slightly faster
- The model without self-loop takes a bit longer to reach peak performance
- Symmetric normalization shows steady improvement throughout training

## Recommendations

1. **Include self-loops**: They provide a consistent accuracy boost
2. **Use left normalization** on citation networks like Cora
3. **Experiment with different normalization techniques** for different datasets
4. **The baseline configuration is robust** and can be used as a starting point for further improvements

## Comparison with Original Paper

- The original GCN paper reports 81.5% accuracy on Cora
- Our baseline achieves 83.5%, which is slightly better
- This may be due to different random seeds or implementation details