"""Model builder factory for creating models from configuration."""
import tensorflow as tf
from tensorflow import keras
from typing import Optional

from ..config import ModelConfig, TrainingConfig
from .baseline_cnn import create_baseline_from_config
from .transfer_models import create_transfer_model_from_config, unfreeze_base_model


def build_model(
    model_config: ModelConfig,
    training_config: Optional[TrainingConfig] = None
) -> keras.Model:
    """Build model from configuration.

    Args:
        model_config: Model configuration
        training_config: Training configuration (optional)

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_config.model_type.lower()

    # Create model based on type
    if model_type == 'baseline':
        model = create_baseline_from_config(model_config)

    elif model_type in ['mobilenet_v2', 'mobilenet', 'efficientnet_b0', 'efficientnet']:
        model = create_transfer_model_from_config(model_config)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: 'baseline', 'mobilenet_v2', 'efficientnet_b0'"
        )

    # Compile model if training config provided
    if training_config is not None:
        model = compile_model(model, training_config)

    return model


def compile_model(
    model: keras.Model,
    training_config: TrainingConfig
) -> keras.Model:
    """Compile model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        training_config: Training configuration

    Returns:
        Compiled model
    """
    # Create optimizer
    optimizer = create_optimizer(training_config)

    # Create loss function
    loss = create_loss_function(training_config)

    # Create metrics
    metrics = create_metrics()

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def create_optimizer(training_config: TrainingConfig) -> keras.optimizers.Optimizer:
    """Create optimizer from training configuration.

    Args:
        training_config: Training configuration

    Returns:
        Keras optimizer

    Raises:
        ValueError: If optimizer type is not supported
    """
    # Create learning rate schedule if enabled
    if training_config.lr_decay_enabled:
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=training_config.learning_rate,
            decay_steps=training_config.lr_decay_steps,
            decay_rate=training_config.lr_decay_rate,
            staircase=True
        )
    else:
        learning_rate = training_config.learning_rate

    # Create optimizer
    optimizer_type = training_config.optimizer.lower()

    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. "
            f"Supported: 'adam', 'sgd', 'rmsprop'"
        )

    return optimizer


def create_loss_function(training_config: TrainingConfig) -> keras.losses.Loss:
    """Create loss function from training configuration.

    Args:
        training_config: Training configuration

    Returns:
        Keras loss function
    """
    # Categorical crossentropy with optional label smoothing
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=training_config.label_smoothing
    )

    return loss


def create_metrics() -> list:
    """Create list of metrics to track during training.

    Returns:
        List of Keras metrics
    """
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    return metrics


def create_callbacks(
    training_config: TrainingConfig,
    checkpoint_path: str = './saved_models/checkpoint.keras',
    log_dir: str = './logs',
    verbose: int = 1
) -> list:
    """Create training callbacks.

    Args:
        training_config: Training configuration
        checkpoint_path: Path to save model checkpoints
        log_dir: Directory for TensorBoard logs
        verbose: Verbosity level

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # Model checkpoint
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=training_config.checkpoint_monitor,
        mode=training_config.checkpoint_mode,
        save_best_only=training_config.save_best_only,
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=training_config.early_stopping_monitor,
        mode=training_config.early_stopping_mode,
        patience=training_config.early_stopping_patience,
        verbose=verbose,
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

    # TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)

    # Learning rate scheduler (if needed)
    if training_config.lr_decay_enabled:
        # Already handled in optimizer, but we can add a callback to log it
        lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_config.lr_decay_rate,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        # Note: We're using exponential decay in optimizer, so this is optional
        # callbacks.append(lr_callback)

    return callbacks


def get_model_summary(model: keras.Model) -> dict:
    """Get model summary information.

    Args:
        model: Keras model

    Returns:
        Dictionary with model information
    """
    return {
        'name': model.name,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


if __name__ == "__main__":
    from ..config import Config

    # Test model building
    config = Config()

    # Test baseline model
    print("Building baseline model...")
    config.model.model_type = "baseline"
    model = build_model(config.model, config.training)
    print(get_model_summary(model))

    # Test MobileNetV2 model
    print("\n\nBuilding MobileNetV2 model...")
    config.model.model_type = "mobilenet_v2"
    model = build_model(config.model, config.training)
    print(get_model_summary(model))
