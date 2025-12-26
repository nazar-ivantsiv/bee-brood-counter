"""Hyperparameter search space definitions for model tuning."""
import keras_tuner as kt
from tensorflow import keras
from typing import Dict, Any

from ..config import ModelConfig


def build_tunable_baseline_cnn(hp: kt.HyperParameters) -> keras.Model:
    """Build baseline CNN with tunable hyperparameters.

    Args:
        hp: Keras Tuner hyperparameters object

    Returns:
        Compiled Keras model
    """
    # Tunable architecture parameters
    conv1_filters = hp.Int('conv1_filters', min_value=8, max_value=32, step=8)
    conv2_filters = hp.Int('conv2_filters', min_value=8, max_value=32, step=8)
    fc_units = hp.Int('fc_units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    # Build model
    inputs = keras.Input(shape=(64, 64, 3), name='input')

    # Conv layer 1
    x = keras.layers.Conv2D(
        filters=conv1_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        name='conv1'
    )(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same', name='pool1')(x)

    # Optional dropout after first pooling
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout1')(x)

    # Conv layer 2
    x = keras.layers.Conv2D(
        filters=conv2_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        name='conv2'
    )(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Flatten
    x = keras.layers.Flatten(name='flatten')(x)

    # Optional dropout before FC
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout2')(x)

    # Fully connected layer
    x = keras.layers.Dense(
        units=fc_units,
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        name='fc1'
    )(x)

    # Output layer
    outputs = keras.layers.Dense(
        units=2,
        activation='softmax',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=keras.initializers.Constant(1.0),
        name='output'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='tunable_baseline_cnn')

    # Tunable training parameters
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3])
    optimizer_type = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])

    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:  # sgd
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )

    return model


def build_tunable_mobilenet_v2(hp: kt.HyperParameters) -> keras.Model:
    """Build MobileNetV2 with tunable hyperparameters.

    Args:
        hp: Keras Tuner hyperparameters object

    Returns:
        Compiled Keras model
    """
    # Load pretrained base
    base_model = keras.applications.MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Start frozen

    # Tunable classifier head
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=2)

    # Build model
    inputs = keras.Input(shape=(64, 64, 3), name='input')
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # First dropout
    x = keras.layers.Dropout(dropout_rate, name='dropout_0')(x)

    # Dense layers
    for i in range(num_dense_layers):
        x = keras.layers.Dense(
            units=dense_units // (2 ** i),  # Decreasing units
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = keras.layers.Dropout(dropout_rate / (1 + i), name=f'dropout_{i+1}')(x)

    # Output layer
    outputs = keras.layers.Dense(2, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='tunable_mobilenet_v2')

    # Tunable training parameters
    learning_rate = hp.Choice('learning_rate', values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    optimizer_type = hp.Choice('optimizer', values=['adam', 'rmsprop'])

    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )

    return model


def get_search_space_summary(model_type: str) -> Dict[str, Any]:
    """Get summary of hyperparameter search space.

    Args:
        model_type: Type of model ('baseline' or 'mobilenet_v2')

    Returns:
        Dictionary describing the search space
    """
    if model_type == 'baseline':
        return {
            'architecture': {
                'conv1_filters': [8, 16, 24, 32],
                'conv2_filters': [8, 16, 24, 32],
                'fc_units': [32, 64, 96, 128],
                'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            },
            'training': {
                'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
                'optimizer': ['adam', 'rmsprop', 'sgd'],
            },
            'total_combinations': 6 * 6 * 4 * 6 * 4 * 3,  # ~10,368 combinations
        }
    elif model_type == 'mobilenet_v2':
        return {
            'architecture': {
                'dense_units': [64, 128, 192, 256],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'num_dense_layers': [1, 2],
            },
            'training': {
                'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                'optimizer': ['adam', 'rmsprop'],
            },
            'total_combinations': 4 * 4 * 2 * 5 * 2,  # ~320 combinations
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
