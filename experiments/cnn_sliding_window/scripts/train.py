#!/usr/bin/env python3
"""Training script for bee brood counter models."""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import create_datasets_from_config, DatasetLoader
from src.models.model_builder import build_model, create_callbacks, get_model_summary
from src.models.transfer_models import unfreeze_base_model


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")


def setup_gpu(gpu_id: int = 0, mixed_precision: bool = False):
    """Setup GPU configuration.

    Args:
        gpu_id: GPU device ID to use
        mixed_precision: Enable mixed precision training
    """
    # Set visible GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Use specific GPU
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

    # Setup mixed precision if requested
    if mixed_precision:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled")


def train_single_stage(
    config: Config,
    output_dir: str = "./saved_models",
    log_dir: str = "./logs",
    use_mlflow: bool = True
) -> keras.Model:
    """Train model in single stage.

    Args:
        config: Configuration object
        output_dir: Directory to save models
        log_dir: Directory for logs
        use_mlflow: Use MLflow for experiment tracking

    Returns:
        Trained model
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Build model
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)
    model = build_model(config.model, config.training)

    # Print model summary
    summary = get_model_summary(model)
    print(f"\nModel: {summary['name']}")
    print(f"Total parameters: {summary['total_params']:,}")
    print(f"Trainable parameters: {summary['trainable_params']:,}")
    print(f"Non-trainable parameters: {summary['non_trainable_params']:,}")

    # Create datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    datasets = create_datasets_from_config(
        config=config.data,
        augmentation_config=config.augmentation,
        batch_size=config.training.batch_size,
        cache=False
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    class_weights = datasets['class_weights']

    print(f"Training samples: {datasets['sizes']['train']}")
    print(f"Validation samples: {datasets['sizes']['val']}")
    print(f"Test samples: {datasets['sizes']['test']}")
    print(f"Class weights: {class_weights}")

    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        output_dir,
        f"{config.model.model_type}_{timestamp}_best.keras"
    )
    tensorboard_log_dir = os.path.join(log_dir, f"{config.model.model_type}_{timestamp}")

    callbacks = create_callbacks(
        training_config=config.training,
        checkpoint_path=checkpoint_path,
        log_dir=tensorboard_log_dir,
        verbose=config.verbose
    )

    # Start MLflow run if requested
    if use_mlflow:
        mlflow.set_tracking_uri(config.experiment.tracking_uri)
        mlflow.set_experiment(config.experiment.experiment_name)

        run_name = config.experiment.run_name or f"{config.model.model_type}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "model_type": config.model.model_type,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.epochs,
                "optimizer": config.training.optimizer,
                "augmentation_enabled": config.augmentation.enabled,
                "use_class_weights": config.training.use_class_weights,
                **summary
            })

            # Train model
            print("\n" + "="*80)
            print("Training model...")
            print("="*80)

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=config.training.epochs,
                callbacks=callbacks,
                class_weight=class_weights if config.training.use_class_weights else None,
                verbose=config.verbose
            )

            # Log metrics
            for epoch, (train_loss, val_loss) in enumerate(
                zip(history.history['loss'], history.history['val_loss'])
            ):
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": history.history['accuracy'][epoch],
                    "val_accuracy": history.history['val_accuracy'][epoch],
                }, step=epoch)

            # Log model
            if config.experiment.log_models:
                mlflow.keras.log_model(model, "model")

            print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")

    else:
        # Train without MLflow
        print("\n" + "="*80)
        print("Training model...")
        print("="*80)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.training.epochs,
            callbacks=callbacks,
            class_weight=class_weights if config.training.use_class_weights else None,
            verbose=config.verbose
        )

    # Save final model
    final_model_path = os.path.join(
        output_dir,
        f"{config.model.model_type}_{timestamp}_final.keras"
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)

    test_dataset = datasets['test']
    test_results = model.evaluate(test_dataset, verbose=config.verbose)

    print(f"\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")

    if use_mlflow:
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
            mlflow.log_metrics({
                f"test_{name}": value
                for name, value in zip(model.metrics_names, test_results)
            })

    return model


def train_two_stage(
    config: Config,
    output_dir: str = "./saved_models",
    log_dir: str = "./logs",
    use_mlflow: bool = True
) -> keras.Model:
    """Train transfer learning model in two stages.

    Stage 1: Train classifier head only (frozen base)
    Stage 2: Fine-tune entire model (unfrozen base)

    Args:
        config: Configuration object
        output_dir: Directory to save models
        log_dir: Directory for logs
        use_mlflow: Use MLflow for experiment tracking

    Returns:
        Trained model
    """
    print("\n" + "="*80)
    print("TWO-STAGE TRAINING FOR TRANSFER LEARNING")
    print("="*80)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Build model with frozen base
    print("\nBuilding model (frozen base)...")
    config.model.freeze_base = True
    model = build_model(config.model, config.training)

    # Create datasets
    print("\nLoading datasets...")
    datasets = create_datasets_from_config(
        config=config.data,
        augmentation_config=config.augmentation,
        batch_size=config.training.batch_size,
        cache=False
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    class_weights = datasets['class_weights']

    print(f"Training samples: {datasets['sizes']['train']}")
    print(f"Class weights: {class_weights}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_mlflow:
        mlflow.set_tracking_uri(config.experiment.tracking_uri)
        mlflow.set_experiment(config.experiment.experiment_name)

        run_name = config.experiment.run_name or f"{config.model.model_type}_{timestamp}_2stage"

        with mlflow.start_run(run_name=run_name):
            # Log common parameters
            mlflow.log_params({
                "model_type": config.model.model_type,
                "batch_size": config.training.batch_size,
                "training_strategy": "two_stage",
                "augmentation_enabled": config.augmentation.enabled,
            })

            # Stage 1: Train classifier head
            print("\n" + "="*80)
            print("STAGE 1: Training classifier head (frozen base)")
            print("="*80)

            # Update learning rate and epochs for stage 1
            stage1_lr = config.training.stage1_lr
            stage1_epochs = config.training.stage1_epochs

            # Recompile with stage 1 learning rate
            optimizer = keras.optimizers.Adam(learning_rate=stage1_lr)
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.CategoricalCrossentropy(
                    label_smoothing=config.training.label_smoothing
                ),
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )

            # Callbacks for stage 1
            checkpoint_path_s1 = os.path.join(
                output_dir,
                f"{config.model.model_type}_{timestamp}_stage1.keras"
            )
            callbacks_s1 = create_callbacks(
                training_config=config.training,
                checkpoint_path=checkpoint_path_s1,
                log_dir=os.path.join(log_dir, f"{config.model.model_type}_{timestamp}_stage1"),
                verbose=config.verbose
            )

            # Train stage 1
            history_s1 = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=stage1_epochs,
                callbacks=callbacks_s1,
                class_weight=class_weights if config.training.use_class_weights else None,
                verbose=config.verbose
            )

            # Log stage 1 metrics
            for epoch in range(stage1_epochs):
                mlflow.log_metrics({
                    "stage1_train_loss": history_s1.history['loss'][epoch],
                    "stage1_val_loss": history_s1.history['val_loss'][epoch],
                    "stage1_train_accuracy": history_s1.history['accuracy'][epoch],
                    "stage1_val_accuracy": history_s1.history['val_accuracy'][epoch],
                }, step=epoch)

            # Stage 2: Fine-tune entire model
            print("\n" + "="*80)
            print("STAGE 2: Fine-tuning entire model (unfrozen base)")
            print("="*80)

            # Unfreeze base model
            model = unfreeze_base_model(model)

            # Update learning rate and epochs for stage 2
            stage2_lr = config.training.stage2_lr
            stage2_epochs = config.training.stage2_epochs

            # Recompile with stage 2 learning rate (much lower)
            optimizer = keras.optimizers.Adam(learning_rate=stage2_lr)
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.CategoricalCrossentropy(
                    label_smoothing=config.training.label_smoothing
                ),
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )

            # Callbacks for stage 2
            checkpoint_path_s2 = os.path.join(
                output_dir,
                f"{config.model.model_type}_{timestamp}_stage2_best.keras"
            )
            callbacks_s2 = create_callbacks(
                training_config=config.training,
                checkpoint_path=checkpoint_path_s2,
                log_dir=os.path.join(log_dir, f"{config.model.model_type}_{timestamp}_stage2"),
                verbose=config.verbose
            )

            # Train stage 2
            history_s2 = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=stage2_epochs,
                callbacks=callbacks_s2,
                class_weight=class_weights if config.training.use_class_weights else None,
                verbose=config.verbose
            )

            # Log stage 2 metrics
            for epoch in range(stage2_epochs):
                mlflow.log_metrics({
                    "stage2_train_loss": history_s2.history['loss'][epoch],
                    "stage2_val_loss": history_s2.history['val_loss'][epoch],
                    "stage2_train_accuracy": history_s2.history['accuracy'][epoch],
                    "stage2_val_accuracy": history_s2.history['val_accuracy'][epoch],
                }, step=epoch)

            # Evaluate on test set
            print("\n" + "="*80)
            print("Evaluating on test set...")
            print("="*80)

            test_dataset = datasets['test']
            test_results = model.evaluate(test_dataset, verbose=config.verbose)

            print(f"\nTest Results:")
            for metric_name, value in zip(model.metrics_names, test_results):
                print(f"  {metric_name}: {value:.4f}")

            mlflow.log_metrics({
                f"test_{name}": value
                for name, value in zip(model.metrics_names, test_results)
            })

            # Log final model
            if config.experiment.log_models:
                mlflow.keras.log_model(model, "model")

            print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")

    # Save final model
    final_model_path = os.path.join(
        output_dir,
        f"{config.model.model_type}_{timestamp}_final.keras"
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train bee brood counter model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset path from config"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training for transfer learning"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./saved_models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # Apply command-line overrides
    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    config.data.seed = args.seed
    config.gpu_id = args.gpu

    # Set seeds for reproducibility
    set_seeds(args.seed)

    # Setup GPU
    setup_gpu(args.gpu, config.mixed_precision)

    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Model type: {config.model.model_type}")
    print(f"Dataset: {config.data.dataset_path}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Augmentation: {config.augmentation.enabled}")
    print(f"Use class weights: {config.training.use_class_weights}")
    print(f"MLflow tracking: {not args.no_mlflow}")
    print(f"Two-stage training: {args.two_stage}")

    # Train model
    if args.two_stage and config.model.model_type != 'baseline':
        model = train_two_stage(
            config=config,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            use_mlflow=not args.no_mlflow
        )
    else:
        model = train_single_stage(
            config=config,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            use_mlflow=not args.no_mlflow
        )

    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
