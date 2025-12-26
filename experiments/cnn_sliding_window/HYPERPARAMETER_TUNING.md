# Hyperparameter Tuning Guide

Complete guide to automatically finding the best hyperparameters for your bee brood counter models.

---

## What is Hyperparameter Tuning?

Hyperparameter tuning automatically searches for the best model configuration by trying different combinations of:
- **Architecture parameters**: Number of filters, layer sizes, dropout rates
- **Training parameters**: Learning rate, optimizer, batch size
- **Regularization**: Dropout, L2 regularization

Instead of manually trying different values, the tuner explores the search space intelligently using the **Hyperband algorithm**.

---

## Quick Start

### 1. Basic Tuning (Baseline CNN)

```bash
source ./bee_brood_counter/bin/activate

# Quick test (10 trials, ~15 minutes)
python scripts/tune_hyperparams.py \
    --model baseline \
    --max-trials 10 \
    --max-epochs 10

# Full tuning (50 trials, ~2-3 hours)
python scripts/tune_hyperparams.py \
    --model baseline \
    --max-trials 50 \
    --max-epochs 30
```

### 2. Tune MobileNetV2

```bash
# Recommended for 8GB M1 Macs (CPU only)
python scripts/tune_hyperparams.py \
    --model mobilenet_v2 \
    --max-trials 30 \
    --max-epochs 20 \
    --cpu-only
```

### 3. View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./experiments

# Open http://localhost:5000 in browser
# Compare all trials and see which hyperparameters work best
```

---

## Search Space

### Baseline CNN Search Space

**Architecture Parameters:**
- `conv1_filters`: [8, 16, 24, 32] - Number of filters in first conv layer
- `conv2_filters`: [8, 16, 24, 32] - Number of filters in second conv layer
- `fc_units`: [32, 64, 96, 128] - Units in fully connected layer
- `dropout_rate`: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] - Dropout rate

**Training Parameters:**
- `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3] - Initial learning rate
- `optimizer`: ['adam', 'rmsprop', 'sgd'] - Optimization algorithm

**Total combinations**: ~10,368

### MobileNetV2 Search Space

**Architecture Parameters:**
- `dense_units`: [64, 128, 192, 256] - Units in classifier head
- `dropout_rate`: [0.2, 0.3, 0.4, 0.5] - Dropout rate
- `num_dense_layers`: [1, 2] - Number of dense layers in head

**Training Parameters:**
- `learning_rate`: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] - Initial learning rate
- `optimizer`: ['adam', 'rmsprop'] - Optimization algorithm

**Total combinations**: ~320

---

## How It Works

### 1. Hyperband Algorithm

The tuner uses **Hyperband**, an efficient algorithm that:
1. Starts with many configurations (trials)
2. Trains each for a few epochs
3. Keeps the best-performing ones
4. Trains those longer
5. Repeats until finding the best

This is much faster than training all configurations to completion!

### 2. Objective Function

The tuner optimizes **validation AUC** (Area Under ROC Curve):
- Better than accuracy for imbalanced data
- Measures model's ability to distinguish classes
- Values range from 0.5 (random) to 1.0 (perfect)

### 3. Early Stopping

Each trial stops early if validation loss doesn't improve for 5 epochs:
- Saves time on poor configurations
- Prevents overfitting
- More trials can be explored

---

## Usage Examples

### Example 1: Quick Test (10 trials)

```bash
python scripts/tune_hyperparams.py \
    --model baseline \
    --max-trials 10 \
    --max-epochs 10 \
    --cpu-only
```

**Output:**
```
HYPERPARAMETER TUNING: BASELINE
================================================================================

SEARCH SPACE
================================================================================
{
  "architecture": {
    "conv1_filters": [8, 16, 24, 32],
    "conv2_filters": [8, 16, 24, 32],
    "fc_units": [32, 64, 96, 128],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  },
  "training": {
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "optimizer": ["adam", "rmsprop", "sgd"]
  },
  "total_combinations": 10368
}

Total possible combinations: 10,368
Trials to run: 10
Coverage: 0.10%

...

Top 5 Hyperparameter Configurations:
--------------------------------------------------------------------------------

#1 Configuration:
  conv1_filters: 24
  conv2_filters: 16
  fc_units: 96
  dropout_rate: 0.2
  learning_rate: 0.001
  optimizer: adam
  Parameters: 65,234

#1 Trial trial_001:
  Score (val_auc): 0.8945
  Best epoch: 8
  Val AUC: 0.8945

✓ Best model saved to: ./hyperparameter_tuning/baseline_20241209_123456/best_model.keras
✓ Best hyperparameters saved to: ./hyperparameter_tuning/baseline_20241209_123456/best_hyperparameters.json
```

### Example 2: Full Baseline Tuning

```bash
python scripts/tune_hyperparams.py \
    --model baseline \
    --max-trials 50 \
    --max-epochs 30 \
    --batch-size 16
```

**Time**: ~2-3 hours on CPU, ~30 minutes on GPU
**Output**: Best model + hyperparameters saved

### Example 3: MobileNetV2 Tuning

```bash
python scripts/tune_hyperparams.py \
    --model mobilenet_v2 \
    --max-trials 30 \
    --max-epochs 20 \
    --cpu-only
```

**Time**: ~3-4 hours on CPU
**Output**: Tuned transfer learning model

---

## Interpreting Results

### 1. MLflow UI

```bash
mlflow ui --backend-store-uri ./experiments
```

In the UI, you can:
- **Compare all trials** side-by-side
- **See which hyperparameters** matter most
- **Filter by metrics** (val_auc, accuracy, etc.)
- **Download best models**

### 2. Best Hyperparameters File

Check `best_hyperparameters.json`:

```json
{
  "conv1_filters": 24,
  "conv2_filters": 16,
  "fc_units": 96,
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "optimizer": "adam"
}
```

These are the best values found!

### 3. Performance Comparison

Typical improvements from tuning:

| Model | Before Tuning | After Tuning | Improvement |
|-------|---------------|--------------|-------------|
| Baseline CNN | ~82% accuracy | ~87-90% accuracy | +5-8% |
| MobileNetV2 | ~88% accuracy | ~92-95% accuracy | +4-7% |

---

## Advanced Usage

### Resume Interrupted Search

If tuning is interrupted (Ctrl+C or crash), just run the same command again:
```bash
python scripts/tune_hyperparams.py --model baseline --max-trials 50
```

The tuner automatically resumes from where it stopped!

### Custom Search Space

Edit `src/training/hyperparameters.py` to modify the search space:

```python
# Example: Try more learning rates
learning_rate = hp.Choice('learning_rate',
                         values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

# Example: Different dropout range
dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.05)
```

### Optimize Different Metrics

Change the objective in `tune_hyperparams.py`:

```python
# Optimize validation accuracy instead of AUC
tuner = kt.Hyperband(
    hypermodel=model_builder,
    objective=kt.Objective('val_accuracy', direction='max'),  # Changed!
    ...
)
```

Options:
- `val_accuracy` - Raw accuracy
- `val_precision` - Minimize false positives
- `val_recall` - Minimize false negatives
- `val_auc` - Overall discrimination ability (recommended)

### Use Best Hyperparameters

After tuning, use the best hyperparameters in your config:

```yaml
# configs/baseline_tuned.yaml
model:
  model_type: "baseline"
  conv1_filters: 24  # From tuning
  conv2_filters: 16  # From tuning
  fc_units: 96       # From tuning

training:
  learning_rate: 0.001  # From tuning
  optimizer: "adam"      # From tuning
```

Then train with these settings:
```bash
python scripts/train.py --config configs/baseline_tuned.yaml --epochs 100
```

---

## Tips & Best Practices

### 1. Start Small

```bash
# Quick sanity check (10 trials, 10 epochs)
python scripts/tune_hyperparams.py --max-trials 10 --max-epochs 10 --cpu-only
```

Make sure everything works before running a full search!

### 2. Use CPU for Stability

On 8GB M1 Macs, use `--cpu-only`:
```bash
python scripts/tune_hyperparams.py --cpu-only
```

Slower but won't crash due to Metal memory issues.

### 3. Run Overnight

Full tuning takes time. Start it before bed:
```bash
nohup python scripts/tune_hyperparams.py --max-trials 50 > tuning.log 2>&1 &
```

Check progress:
```bash
tail -f tuning.log
```

### 4. Tune in Stages

1. **Stage 1**: Tune architecture (10-20 trials)
2. **Stage 2**: Fix architecture, tune learning rate (10-20 trials)
3. **Stage 3**: Fine-tune everything (20-30 trials)

### 5. Watch for Overfitting

If `val_auc` is much lower than `train_auc`:
- Increase dropout
- Add more regularization
- Reduce model complexity

---

## Troubleshooting

### Issue: "Out of memory" during tuning

**Solution:**
```bash
# Reduce batch size
python scripts/tune_hyperparams.py --batch-size 8 --cpu-only

# Or reduce max trials
python scripts/tune_hyperparams.py --max-trials 20
```

### Issue: Tuning takes too long

**Solution:**
```bash
# Reduce max epochs per trial
python scripts/tune_hyperparams.py --max-epochs 15

# Reduce number of trials
python scripts/tune_hyperparams.py --max-trials 30
```

### Issue: All trials perform poorly

**Possible causes:**
1. **Data issue** - Check dataset is loaded correctly
2. **Search space too narrow** - Widen the range of hyperparameters
3. **Wrong objective** - Try optimizing different metric

### Issue: Results not in MLflow

Make sure MLflow UI is pointing to correct directory:
```bash
mlflow ui --backend-store-uri ./experiments
```

---

## Expected Results

### Baseline CNN (50 trials, 30 epochs)

**Before tuning:**
- Validation accuracy: ~82%
- Validation AUC: ~0.85

**After tuning:**
- Validation accuracy: ~87-90%
- Validation AUC: ~0.90-0.92

**Best hyperparameters typically:**
- conv1_filters: 16-24
- conv2_filters: 16-24
- fc_units: 64-96
- dropout_rate: 0.1-0.3
- learning_rate: 1e-3 to 5e-3
- optimizer: adam

### MobileNetV2 (30 trials, 20 epochs)

**Before tuning:**
- Validation accuracy: ~88%
- Validation AUC: ~0.89

**After tuning:**
- Validation accuracy: ~92-95%
- Validation AUC: ~0.93-0.96

**Best hyperparameters typically:**
- dense_units: 128-192
- dropout_rate: 0.3-0.4
- num_dense_layers: 1-2
- learning_rate: 1e-4 to 5e-4
- optimizer: adam

---

## Next Steps

After tuning:

1. **Train final model** with best hyperparameters (100+ epochs)
2. **Evaluate on test set** to verify performance
3. **Compare with baseline** to measure improvement
4. **Save best model** for deployment
5. **Document results** in your README

---

## Summary

Hyperparameter tuning is essential for:
- ✅ Finding optimal model configuration
- ✅ Maximizing accuracy on your specific dataset
- ✅ Understanding which parameters matter most
- ✅ Avoiding manual trial-and-error

**Time investment**: 2-4 hours of compute
**Performance gain**: Typically 5-8% accuracy improvement
**Automation level**: Fully automated, just start and wait!

🎯 **Recommended workflow:**
1. Run quick test (10 trials) to verify setup
2. Run full tuning overnight (50 trials)
3. Use best hyperparameters for final training
4. Achieve 90%+ accuracy on bee brood detection!
