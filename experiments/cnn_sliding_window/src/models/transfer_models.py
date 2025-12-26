"""Transfer learning models for bee brood detection."""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List

from ..config import ModelConfig


def create_mobilenet_v2(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 2,
    freeze_base: bool = True,
    dense_units: List[int] = [128],
    dropout_rates: List[float] = [0.3, 0.2],
    l2_reg: float = 0.0
) -> keras.Model:
    """Create MobileNetV2-based transfer learning model.

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        freeze_base: Freeze base model weights (for stage 1 training)
        dense_units: List of units for dense layers in classifier head
        dropout_rates: List of dropout rates (should be len(dense_units) + 1)
        l2_reg: L2 regularization strength

    Returns:
        Keras Model
    """
    # Load pretrained MobileNetV2 (without top classification layer)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model if requested
    base_model.trainable = not freeze_base

    # Regularizer
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Build classifier head
    inputs = keras.Input(shape=input_shape, name='input')
    x = base_model(inputs, training=False if freeze_base else True)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Add dropout
    if len(dropout_rates) > 0:
        x = keras.layers.Dropout(dropout_rates[0], name='dropout_0')(x)

    # Add dense layers
    for i, units in enumerate(dense_units):
        x = keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizer,
            name=f'dense_{i+1}'
        )(x)

        # Add dropout after each dense layer
        if i + 1 < len(dropout_rates):
            x = keras.layers.Dropout(
                dropout_rates[i + 1],
                name=f'dropout_{i+1}'
            )(x)

    # Output layer
    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='mobilenet_v2')

    return model


def create_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 2,
    freeze_base: bool = True,
    dense_units: List[int] = [256, 128],
    dropout_rates: List[float] = [0.4, 0.3, 0.2],
    l2_reg: float = 0.0
) -> keras.Model:
    """Create EfficientNetB0-based transfer learning model.

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        freeze_base: Freeze base model weights (for stage 1 training)
        dense_units: List of units for dense layers in classifier head
        dropout_rates: List of dropout rates (should be len(dense_units) + 1)
        l2_reg: L2 regularization strength

    Returns:
        Keras Model
    """
    # Load pretrained EfficientNetB0 (without top classification layer)
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model if requested
    base_model.trainable = not freeze_base

    # Regularizer
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Build classifier head
    inputs = keras.Input(shape=input_shape, name='input')
    x = base_model(inputs, training=False if freeze_base else True)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Add dropout
    if len(dropout_rates) > 0:
        x = keras.layers.Dropout(dropout_rates[0], name='dropout_0')(x)

    # Add dense layers
    for i, units in enumerate(dense_units):
        x = keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizer,
            name=f'dense_{i+1}'
        )(x)

        # Add dropout after each dense layer
        if i + 1 < len(dropout_rates):
            x = keras.layers.Dropout(
                dropout_rates[i + 1],
                name=f'dropout_{i+1}'
            )(x)

    # Output layer
    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_b0')

    return model


def unfreeze_base_model(model: keras.Model) -> keras.Model:
    """Unfreeze base model for fine-tuning (stage 2 training).

    Args:
        model: Model with frozen base

    Returns:
        Model with unfrozen base
    """
    # Find the base model layer
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # This is the base model
            layer.trainable = True

    return model


def create_transfer_model_from_config(config: ModelConfig) -> keras.Model:
    """Create transfer learning model from configuration.

    Args:
        config: Model configuration

    Returns:
        Keras Model

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = config.model_type.lower()

    if model_type == 'mobilenet_v2' or model_type == 'mobilenet':
        return create_mobilenet_v2(
            input_shape=tuple(config.input_shape),
            num_classes=config.num_classes,
            freeze_base=config.freeze_base,
            dense_units=config.dense_units,
            dropout_rates=config.dropout_rates,
            l2_reg=0.0  # Set in training config
        )

    elif model_type == 'efficientnet_b0' or model_type == 'efficientnet':
        return create_efficientnet_b0(
            input_shape=tuple(config.input_shape),
            num_classes=config.num_classes,
            freeze_base=config.freeze_base,
            dense_units=config.dense_units,
            dropout_rates=config.dropout_rates,
            l2_reg=0.0  # Set in training config
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: 'mobilenet_v2', 'efficientnet_b0'"
        )


if __name__ == "__main__":
    # Test MobileNetV2
    print("Creating MobileNetV2 model...")
    model_mn = create_mobilenet_v2()
    model_mn.summary()

    # Test EfficientNetB0
    print("\n\nCreating EfficientNetB0 model...")
    model_eff = create_efficientnet_b0()
    model_eff.summary()

    # Test forward pass
    import numpy as np
    test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)

    print("\n\nTesting MobileNetV2 forward pass...")
    output_mn = model_mn(test_input)
    print(f"Output shape: {output_mn.shape}")
    print(f"Output probabilities: {output_mn.numpy()[0]}")

    print("\n\nTesting EfficientNetB0 forward pass...")
    output_eff = model_eff(test_input)
    print(f"Output shape: {output_eff.shape}")
    print(f"Output probabilities: {output_eff.numpy()[0]}")
