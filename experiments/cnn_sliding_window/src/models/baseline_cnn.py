"""Baseline CNN model - original 4-layer ConvNet architecture."""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from ..config import ModelConfig


def create_baseline_cnn(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 2,
    conv1_filters: int = 16,
    conv2_filters: int = 16,
    fc_units: int = 64,
    l2_reg: float = 0.0
) -> keras.Model:
    """Create baseline CNN model (original architecture).

    Architecture:
    - CONV1 (16 filters, 5x5) → RELU → MAXPOOL (4x4)
    - CONV2 (16 filters, 5x5) → RELU → MAXPOOL (2x2)
    - FLATTEN
    - FC1 (64 units) → RELU
    - FC2 (2 units) → SOFTMAX

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        conv1_filters: Number of filters in first conv layer
        conv2_filters: Number of filters in second conv layer
        fc_units: Number of units in fully connected layer
        l2_reg: L2 regularization strength

    Returns:
        Keras Model
    """
    # Regularizer
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input')

    # Layer 1: Conv2D + ReLU + MaxPool
    x = keras.layers.Conv2D(
        filters=conv1_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        kernel_regularizer=regularizer,
        name='conv1'
    )(inputs)
    x = keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding='same',
        name='pool1'
    )(x)

    # Layer 2: Conv2D + ReLU + MaxPool
    x = keras.layers.Conv2D(
        filters=conv2_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        kernel_regularizer=regularizer,
        name='conv2'
    )(x)
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='pool2'
    )(x)

    # Flatten
    x = keras.layers.Flatten(name='flatten')(x)

    # Layer 3: Fully Connected + ReLU
    x = keras.layers.Dense(
        units=fc_units,
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        kernel_regularizer=regularizer,
        name='fc1'
    )(x)

    # Layer 4: Output layer (Softmax)
    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        name='output'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='baseline_cnn')

    return model


def create_baseline_from_config(config: ModelConfig) -> keras.Model:
    """Create baseline CNN from configuration.

    Args:
        config: Model configuration

    Returns:
        Keras Model
    """
    return create_baseline_cnn(
        input_shape=tuple(config.input_shape),
        num_classes=config.num_classes,
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters,
        fc_units=config.fc_units,
        l2_reg=0.0  # Will be set in training config
    )


if __name__ == "__main__":
    # Test model creation
    model = create_baseline_cnn()
    model.summary()

    # Test forward pass
    import numpy as np
    test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)
    output = model(test_input)
    print(f"\nTest output shape: {output.shape}")
    print(f"Output probabilities: {output.numpy()[0]}")
