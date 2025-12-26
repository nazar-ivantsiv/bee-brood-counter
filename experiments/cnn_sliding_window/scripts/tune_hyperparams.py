#!/usr/bin/env python3
"""Hyperparameter tuning script using Keras Tuner and MLflow."""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import mlflow
import mlflow.keras

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import create_datasets_from_config
from src.training.hyperparameters import (
    build_tunable_baseline_cnn,
    build_tunable_mobilenet_v2,
    get_search_space_summary
)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class MLflowTunerCallback(keras.callbacks.Callback):
    """Callback to log metrics to MLflow during tuning."""

    def __init__(self, trial_id: str):
        super().__init__()
        self.trial_id = trial_id

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        if logs:
            mlflow.log_metrics(logs, step=epoch)


def tune_hyperparameters(
    model_type: str = 'baseline',
    max_trials: int = 50,
    executions_per_trial: int = 1,
    max_epochs: int = 30,
    batch_size: int = 16,
    dataset_path: str = './dataset',
    tuning_dir: str = './hyperparameter_tuning',
    use_mlflow: bool = True,
    experiment_name: str = 'hyperparameter_tuning',
    cpu_only: bool = False,
    seed: int = 42
):
    """Run hyperparameter tuning.

    Args:
        model_type: Model to tune ('baseline' or 'mobilenet_v2')
        max_trials: Maximum number of trials to run
        executions_per_trial: Number of times to train each configuration
        max_epochs: Maximum epochs per trial
        batch_size: Batch size for training
        dataset_path: Path to dataset
        tuning_dir: Directory for tuning results
        use_mlflow: Use MLflow for tracking
        experiment_name: MLflow experiment name
        cpu_only: Force CPU-only training
        seed: Random seed
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
    print("="*80)

    # Set seeds
    set_seeds(seed)

    # Force CPU if requested
    if cpu_only:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only for stability")

    # Create tuning directory
    os.makedirs(tuning_dir, exist_ok=True)
    project_dir = os.path.join(tuning_dir, f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Load config
    config_file = f'configs/{model_type}.yaml' if model_type == 'baseline' else 'configs/mobilenet.yaml'
    config = Config.from_yaml(config_file)
    config.data.dataset_path = dataset_path
    config.training.batch_size = batch_size

    # Print search space
    print("\n" + "="*80)
    print("SEARCH SPACE")
    print("="*80)
    search_space = get_search_space_summary(model_type)
    print(json.dumps(search_space, indent=2))
    print(f"\nTotal possible combinations: {search_space['total_combinations']:,}")
    print(f"Trials to run: {max_trials}")
    print(f"Coverage: {(max_trials / search_space['total_combinations'] * 100):.2f}%")

    # Create datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    datasets = create_datasets_from_config(
        config=config.data,
        augmentation_config=config.augmentation,
        batch_size=batch_size,
        cache=False
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    class_weights = datasets['class_weights']

    print(f"Training samples: {datasets['sizes']['train']}")
    print(f"Validation samples: {datasets['sizes']['val']}")
    print(f"Class weights: {class_weights}")

    # Select model builder
    if model_type == 'baseline':
        model_builder = build_tunable_baseline_cnn
    elif model_type == 'mobilenet_v2':
        model_builder = build_tunable_mobilenet_v2
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Setup MLflow
    if use_mlflow:
        mlflow.set_tracking_uri(config.experiment.tracking_uri)
        mlflow.set_experiment(experiment_name)

    # Create tuner
    print("\n" + "="*80)
    print("CREATING TUNER")
    print("="*80)

    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective=kt.Objective('val_auc', direction='max'),  # Optimize AUC
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=2,
        executions_per_trial=executions_per_trial,
        directory=tuning_dir,
        project_name=f"{model_type}_tuning",
        overwrite=False,
        seed=seed
    )

    print(f"✓ Tuner created: {tuner.__class__.__name__}")
    print(f"  Objective: {tuner.oracle.objective}")
    print(f"  Max trials: {max_trials}")
    print(f"  Max epochs per trial: {max_epochs}")
    print(f"  Executions per trial: {executions_per_trial}")

    # Custom callback to stop early if needed
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Run search
    print("\n" + "="*80)
    print("STARTING HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Estimated time: {max_trials * max_epochs * 15 / 60:.1f} minutes")
    print("This will run in the background. Results will be saved to:")
    print(f"  {project_dir}")
    print("\nPress Ctrl+C to stop early (results will still be saved)")
    print("-" * 80)

    try:
        if use_mlflow:
            with mlflow.start_run(run_name=f"{model_type}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log tuning parameters
                mlflow.log_params({
                    'model_type': model_type,
                    'max_trials': max_trials,
                    'max_epochs': max_epochs,
                    'batch_size': batch_size,
                    'tuner': 'Hyperband',
                    'objective': 'val_auc',
                })

                # Run search
                tuner.search(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=max_epochs,
                    callbacks=[stop_early],
                    class_weight=class_weights if config.training.use_class_weights else None,
                    verbose=1
                )

                # Log best trial
                best_hp = tuner.get_best_hyperparameters(1)[0]
                mlflow.log_params({f'best_{k}': v for k, v in best_hp.values.items()})

        else:
            # Run without MLflow
            tuner.search(
                train_dataset,
                validation_data=val_dataset,
                epochs=max_epochs,
                callbacks=[stop_early],
                class_weight=class_weights if config.training.use_class_weights else None,
                verbose=1
            )

    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user")
        print("Partial results have been saved")

    # Display results
    print("\n" + "="*80)
    print("TUNING RESULTS")
    print("="*80)

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=5)

    print(f"\nTop 5 Hyperparameter Configurations:")
    print("-" * 80)

    for i, hp in enumerate(best_hps, 1):
        print(f"\n#{i} Configuration:")
        for key, value in hp.values.items():
            print(f"  {key}: {value}")

        # Get best model for this config
        models = tuner.get_best_models(num_models=i)
        if len(models) >= i:
            model = models[i-1]
            print(f"  Parameters: {model.count_params():,}")

    # Get best trials
    best_trials = tuner.oracle.get_best_trials(num_trials=5)

    print(f"\n\nTop 5 Trial Results:")
    print("-" * 80)

    for i, trial in enumerate(best_trials, 1):
        print(f"\n#{i} Trial {trial.trial_id}:")
        print(f"  Score (val_auc): {trial.score:.4f}")
        print(f"  Status: {trial.status}")

        # Print metrics
        if trial.metrics:
            best_step = trial.metrics.get_best_step('val_auc')
            if best_step:
                metrics = trial.metrics.get_history('val_auc')
                if metrics:
                    print(f"  Best epoch: {best_step}")
                    print(f"  Val AUC: {metrics[best_step]:.4f}")

    # Save best model
    print("\n" + "="*80)
    print("SAVING BEST MODEL")
    print("="*80)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model_path = os.path.join(project_dir, 'best_model.keras')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    best_model.save(best_model_path)
    print(f"✓ Best model saved to: {best_model_path}")

    # Save best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_hp_path = os.path.join(project_dir, 'best_hyperparameters.json')
    with open(best_hp_path, 'w') as f:
        json.dump(best_hp.values, f, indent=2)
    print(f"✓ Best hyperparameters saved to: {best_hp_path}")

    # Evaluate best model on test set
    print("\n" + "="*80)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*80)

    test_dataset = datasets['test']
    test_results = best_model.evaluate(test_dataset, verbose=0)

    print("\nTest Set Results:")
    for metric_name, value in zip(best_model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")

    if use_mlflow:
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
            mlflow.log_metrics({
                f'best_test_{name}': value
                for name, value in zip(best_model.metrics_names, test_results)
            })

    # Summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("="*80)
    print(f"\n✓ Trials completed: {len(tuner.oracle.trials)}")
    print(f"✓ Best val_auc: {best_trials[0].score:.4f}")
    print(f"✓ Test accuracy: {test_results[1]:.4f}")
    print(f"✓ Results saved to: {project_dir}")

    if use_mlflow:
        print(f"\n✓ View results in MLflow UI:")
        print(f"  mlflow ui --backend-store-uri {config.experiment.tracking_uri}")
        print(f"  Then open: http://localhost:5000")

    return tuner, best_model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for bee brood counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune baseline CNN (50 trials)
  python scripts/tune_hyperparams.py --model baseline --max-trials 50

  # Tune MobileNetV2 (30 trials, CPU only)
  python scripts/tune_hyperparams.py --model mobilenet_v2 --max-trials 30 --cpu-only

  # Quick test (10 trials, 10 epochs)
  python scripts/tune_hyperparams.py --model baseline --max-trials 10 --max-epochs 10
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='baseline',
        choices=['baseline', 'mobilenet_v2'],
        help='Model to tune'
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=50,
        help='Maximum number of trials (default: 50)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=30,
        help='Maximum epochs per trial (default: 30)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./dataset',
        help='Path to dataset'
    )
    parser.add_argument(
        '--tuning-dir',
        type=str,
        default='./hyperparameter_tuning',
        help='Directory for tuning results'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='hyperparameter_tuning',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only training (recommended for 8GB M1 Macs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    try:
        tuner, best_model = tune_hyperparameters(
            model_type=args.model,
            max_trials=args.max_trials,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path,
            tuning_dir=args.tuning_dir,
            use_mlflow=not args.no_mlflow,
            experiment_name=args.experiment_name,
            cpu_only=args.cpu_only,
            seed=args.seed
        )

        return 0

    except Exception as e:
        print(f"\n✗ Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
