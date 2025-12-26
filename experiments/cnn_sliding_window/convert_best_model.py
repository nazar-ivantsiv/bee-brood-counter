#!/usr/bin/env python3
"""Convert best trial checkpoint to .keras model for evaluation."""
import os
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Add parent directory
sys.path.insert(0, '.')
from src.training.hyperparameters import build_tunable_baseline_cnn, build_tunable_mobilenet_v2
import keras_tuner as kt

# Best baseline trial
trial_dir = './hyperparameter_tuning/baseline_tuning/trial_0008'
output_path = './best_baseline_model.keras'

print(f"Loading trial from: {trial_dir}")

# Read trial hyperparameters
with open(f'{trial_dir}/trial.json') as f:
    trial_data = json.load(f)
    
hp_values = trial_data['hyperparameters']['values']
score = trial_data['score']

print(f"\nTrial score (val_auc): {score:.4f}")
print(f"Hyperparameters: {hp_values}")

# Create hyperparameter object and set values
hp = kt.HyperParameters()
for key, value in hp_values.items():
    if key.startswith('tuner/'):
        continue
    if isinstance(value, int):
        hp.Int(key, min_value=value, max_value=value)
    elif isinstance(value, float):
        hp.Float(key, min_value=value, max_value=value)
    else:
        hp.Choice(key, values=[value])

# Build model with these hyperparameters
print("\nBuilding model...")
model = build_tunable_baseline_cnn(hp)

# Load weights from checkpoint
checkpoint_path = f'{trial_dir}/checkpoint'
print(f"Loading weights from: {checkpoint_path}")
model.load_weights(checkpoint_path)

# Save as .keras file
print(f"\nSaving model to: {output_path}")
model.save(output_path)

print(f"✓ Model saved successfully!")
print(f"\nYou can now evaluate with:")
print(f"  python scripts/evaluate.py --model-path {output_path} --cpu-only")
