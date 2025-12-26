#!/usr/bin/env python3
"""
Quick training test - runs 2 epochs to verify end-to-end pipeline.

This script tests that the entire training pipeline works without
running a full training session.
"""
import sys
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.data.dataset import create_datasets_from_config
from src.models.model_builder import build_model


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def quick_train_test(model_type='baseline', epochs=2):
    """Run a quick training test.

    Args:
        model_type: Model type to test ('baseline', 'mobilenet_v2', 'efficientnet_b0')
        epochs: Number of epochs to train (default: 2)
    """
    print("\n" + "="*70)
    print(f"QUICK TRAINING TEST: {model_type.upper()}")
    print("="*70)

    set_seeds(42)

    # Load configuration
    if model_type == 'baseline':
        config_file = 'configs/baseline.yaml'
    elif model_type == 'mobilenet_v2':
        config_file = 'configs/mobilenet.yaml'
    elif model_type == 'efficientnet_b0':
        config_file = 'configs/efficientnet.yaml'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\n1. Loading config from {config_file}...")
    config = Config.from_yaml(config_file)
    config.training.epochs = epochs
    config.training.batch_size = 16  # Smaller batch for quick test

    # Build model
    print(f"\n2. Building {model_type} model...")
    model = build_model(config.model, config.training)
    print(f"   Model: {model.name}")
    print(f"   Parameters: {model.count_params():,}")

    # Create datasets
    print("\n3. Creating datasets...")
    datasets = create_datasets_from_config(
        config=config.data,
        augmentation_config=config.augmentation,
        batch_size=config.training.batch_size,
        cache=False
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    class_weights = datasets['class_weights']

    print(f"   Training samples: {datasets['sizes']['train']}")
    print(f"   Validation samples: {datasets['sizes']['val']}")
    print(f"   Class weights: {class_weights}")

    # Train for a few epochs
    print(f"\n4. Training for {epochs} epochs...")
    print("-" * 70)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights if config.training.use_class_weights else None,
        verbose=1
    )

    print("-" * 70)

    # Evaluate
    print("\n5. Evaluating on validation set...")
    val_results = model.evaluate(val_dataset, verbose=0)

    print("\nValidation Results:")
    for metric_name, value in zip(model.metrics_names, val_results):
        print(f"  {metric_name}: {value:.4f}")

    # Test prediction
    print("\n6. Testing prediction on single batch...")
    for images, labels in val_dataset.take(1):
        predictions = model.predict(images, verbose=0)
        print(f"   Batch shape: {images.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Sample prediction: {predictions[0]}")

    print("\n" + "="*70)
    print(f"✓ Quick training test completed successfully for {model_type}!")
    print("="*70)

    return True


def main():
    """Run quick training tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Quick training test")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "mobilenet_v2", "efficientnet_b0", "all"],
        help="Model to test (or 'all' for all models)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train"
    )

    args = parser.parse_args()

    try:
        if args.model == "all":
            models = ["baseline", "mobilenet_v2", "efficientnet_b0"]
            for model_type in models:
                try:
                    quick_train_test(model_type, args.epochs)
                except Exception as e:
                    print(f"\n✗ Test failed for {model_type}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            quick_train_test(args.model, args.epochs)

        print("\n🎉 All quick training tests passed!")
        return 0

    except Exception as e:
        print(f"\n✗ Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
