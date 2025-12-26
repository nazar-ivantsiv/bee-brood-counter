# Testing Guide

This guide will help you test the bee brood counter implementation to ensure everything is working correctly.

## Prerequisites

### 1. Install Dependencies

First, install all required packages:

```bash
pip install -r requirements.txt
```

**Note:** Installing TensorFlow and other ML libraries may take a few minutes.

### 2. Verify Dataset

Ensure your dataset is in the correct location:

```
dataset/
├── positive/    # Should contain ~631 PNG images
└── negative/    # Should contain ~2,727 PNG images
```

If you have `dataset.zip`, extract it:

```bash
unzip dataset.zip -d ./
```

---

## Test Suite Overview

We've created three test scripts with increasing complexity:

1. **`test_implementation.py`** - Comprehensive unit tests for all components
2. **`quick_train_test.py`** - Quick end-to-end training test (2 epochs)
3. **`scripts/train.py`** - Full training script

---

## Step 1: Run Comprehensive Tests

This tests all components without training:

```bash
python test_implementation.py
```

### Expected Output:

```
==================================================================
BEE BROOD COUNTER - IMPLEMENTATION TEST SUITE
==================================================================

==================================================================
TEST 1: Module Imports
==================================================================
✓ Config
✓ Preprocessing
✓ Augmentation
✓ Dataset
✓ Baseline CNN
✓ Transfer Models
✓ Model Builder

Passed: 7/7

==================================================================
TEST 2: Configuration Loading
==================================================================
✓ Default configuration
✓ Load from configs/baseline.yaml
✓ Load from configs/mobilenet.yaml
✓ Load from configs/efficientnet.yaml

Passed: 4/4

... (more tests)

==================================================================
TEST SUMMARY
==================================================================
✓ PASS: Imports
✓ PASS: Configuration
✓ PASS: Dataset Paths
✓ PASS: Data Pipeline
✓ PASS: Model Creation
✓ PASS: Augmentation

==================================================================
TOTAL: 6/6 test suites passed
==================================================================

🎉 All tests passed! Implementation is ready to use.
```

### What This Tests:

- ✅ All modules can be imported
- ✅ Configuration files are valid
- ✅ Dataset paths exist and contain images
- ✅ Data loading and preprocessing work
- ✅ All model architectures can be created
- ✅ Data augmentation pipeline works

---

## Step 2: Quick Training Test

Run a quick 2-epoch training to verify the end-to-end pipeline:

### Test Baseline CNN

```bash
python quick_train_test.py --model baseline --epochs 2
```

### Test MobileNetV2

```bash
python quick_train_test.py --model mobilenet_v2 --epochs 2
```

### Test All Models

```bash
python quick_train_test.py --model all --epochs 2
```

### Expected Output:

```
==================================================================
QUICK TRAINING TEST: BASELINE
==================================================================

1. Loading config from configs/baseline.yaml...

2. Building baseline model...
   Model: baseline_cnn
   Parameters: 41,410

3. Creating datasets...
   Training samples: 2004
   Validation samples: 668
   Class weights: {0: 1.0, 1: 4.324324324324325}

4. Training for 2 epochs...
----------------------------------------------------------------------
Epoch 1/2
126/126 [==============================] - 15s 115ms/step - loss: 0.6234 - accuracy: 0.7634 - precision: 0.5123 - recall: 0.6892 - auc: 0.8234 - val_loss: 0.4892 - val_accuracy: 0.8123 - val_precision: 0.6234 - val_recall: 0.7456 - val_auc: 0.8567
Epoch 2/2
126/126 [==============================] - 14s 112ms/step - loss: 0.4567 - accuracy: 0.8123 - precision: 0.6234 - recall: 0.7456 - auc: 0.8892 - val_loss: 0.4234 - val_accuracy: 0.8345 - val_precision: 0.6567 - val_recall: 0.7678 - val_auc: 0.8923
----------------------------------------------------------------------

5. Evaluating on validation set...

Validation Results:
  loss: 0.4234
  accuracy: 0.8345
  precision: 0.6567
  recall: 0.7678
  auc: 0.8923

6. Testing prediction on single batch...
   Batch shape: (16, 64, 64, 3)
   Predictions shape: (16, 2)
   Sample prediction: [0.2134 0.7866]

==================================================================
✓ Quick training test completed successfully for baseline!
==================================================================
```

### What This Tests:

- ✅ Complete training pipeline works
- ✅ Data loading and batching
- ✅ Model compilation and training
- ✅ Loss decreases over epochs
- ✅ Validation evaluation works
- ✅ Predictions can be made

**Time:** ~30 seconds for baseline, ~1-2 minutes for transfer models

---

## Step 3: Test Full Training (Optional)

If quick tests pass, try a full training run:

### Baseline CNN (50 epochs)

```bash
python scripts/train.py --config configs/baseline.yaml
```

### MobileNetV2 with Two-Stage Training

```bash
python scripts/train.py \
    --config configs/mobilenet.yaml \
    --two-stage \
    --epochs 50
```

### Monitor with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 in your browser to see live training metrics.

### Monitor with MLflow

In a separate terminal:

```bash
mlflow ui --backend-store-uri ./experiments
```

Open http://localhost:5000 to see experiment tracking.

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.15.0
```

On Apple Silicon Macs, you may need:
```bash
pip install tensorflow-macos tensorflow-metal
```

### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
```

### Issue: "Dataset not found"

**Solution:**
Ensure dataset is extracted:
```bash
unzip dataset.zip -d ./
ls dataset/  # Should show 'positive' and 'negative' directories
```

### Issue: "Out of memory" during training

**Solutions:**
1. Reduce batch size in config file
2. Reduce image size (currently 64x64)
3. Use CPU instead of GPU (slower but uses system RAM)

### Issue: Models train but accuracy is low after 2 epochs

**This is expected!** The quick test only runs 2 epochs to verify the pipeline works.
For actual training, you need 30-50 epochs. The full training script handles this.

---

## Performance Benchmarks

Expected performance after quick test (2 epochs):

| Model | Accuracy | Training Time (2 epochs) |
|-------|----------|--------------------------|
| Baseline CNN | ~70-80% | ~30 seconds |
| MobileNetV2 | ~75-85% | ~1-2 minutes |
| EfficientNetB0 | ~75-85% | ~2-3 minutes |

Expected performance after full training (50 epochs):

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Baseline CNN | ~85-90% | ~0.80-0.85 | ~10-15 minutes |
| MobileNetV2 | ~92-96% | ~0.88-0.92 | ~20-30 minutes |
| EfficientNetB0 | ~94-97% | ~0.90-0.94 | ~30-45 minutes |

*Times are approximate and depend on hardware (CPU vs GPU).*

---

## Next Steps

Once all tests pass:

1. **Train models with full epochs**
2. **Compare models using MLflow UI**
3. **Tune hyperparameters** (Week 5 feature)
4. **Run cross-validation** (Week 6 feature)
5. **Create evaluation reports** (Week 6 feature)

---

## Test Checklist

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset extracted and verified
- [ ] Configuration tests pass (`test_implementation.py`)
- [ ] Quick training test passes for baseline
- [ ] Quick training test passes for MobileNetV2 (optional)
- [ ] Full training completes successfully (optional)
- [ ] MLflow UI shows experiment data (optional)
- [ ] TensorBoard shows training curves (optional)

---

## Getting Help

If tests fail:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Check dataset is in correct location
4. Review the "Troubleshooting" section above
5. Check Python version (requires 3.8+)

The implementation has been designed to provide clear error messages. Most issues are related to:
- Missing dependencies
- Incorrect dataset path
- Insufficient memory (reduce batch size)
