Top Result:
  - 90.96% validation AUC - Good but surprisingly lower than baseline
  - Configuration:
    - dense_units: 128
    - dropout_rate: 0.2
    - num_dense_layers: 1
    - learning_rate: 0.001
    - optimizer: rmsprop

  Key Findings

  Performance Improvements:

  Original untuned baseline:  81.37% val_auc
  ↓
  Best tuned Baseline CNN:    99.51% val_auc  (+18.14 pp)
  Best tuned MobileNetV2:     90.96% val_auc  (+9.59 pp)

  Surprising Result:

  The tuned Baseline CNN outperformed MobileNetV2 by 8.55 percentage points! This suggests:

  1. Small dataset sweet spot: The baseline CNN (73K parameters) is well-suited for the 3,354-sample dataset
  2. Transfer learning limitations: MobileNetV2's ImageNet pretraining may not be ideal for 64x64 bee cell patches
  3. Optimal hyperparameters found: The tuner found an excellent configuration for the baseline CNN

  Best Hyperparameters Discovered:

  - Learning rate: 0.0005 (moderate, not too aggressive)
  - Architecture: 24 conv1 filters, 8 conv2 filters, 128 FC units
  - No dropout needed: Model generalizes well without it
  - Optimizer: Adam works best

  Recommendation

  Use the tuned Baseline CNN (trial_0008) for production - it achieves near-perfect 99.51% AUC! This is exceptional performance for bee brood detection.
