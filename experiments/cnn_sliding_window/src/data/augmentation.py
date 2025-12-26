"""Data augmentation pipeline for bee brood detection."""
import tensorflow as tf
from typing import Optional

from ..config import AugmentationConfig


def create_augmentation_layer(config: AugmentationConfig) -> tf.keras.Sequential:
    """Create data augmentation layer from configuration.

    Args:
        config: Augmentation configuration

    Returns:
        Sequential model with augmentation layers
    """
    if not config.enabled:
        return tf.keras.Sequential([])  # No augmentation

    layers = []

    # Random rotation (full 360° for bee cells)
    if config.rotation_factor > 0:
        layers.append(
            tf.keras.layers.RandomRotation(
                factor=config.rotation_factor,
                fill_mode='reflect',
                interpolation='bilinear'
            )
        )

    # Random flip
    if config.flip_mode:
        layers.append(
            tf.keras.layers.RandomFlip(mode=config.flip_mode)
        )

    # Random zoom
    if config.zoom_range[0] != 0 or config.zoom_range[1] != 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=config.zoom_range,
                width_factor=config.zoom_range,
                fill_mode='reflect'
            )
        )

    # Random brightness
    if config.brightness_factor > 0:
        layers.append(
            tf.keras.layers.RandomBrightness(
                factor=config.brightness_factor,
                value_range=(-0.5, 0.5)  # Match normalized range
            )
        )

    # Random contrast
    if config.contrast_factor > 0:
        layers.append(
            tf.keras.layers.RandomContrast(
                factor=config.contrast_factor
            )
        )

    # Gaussian noise
    if config.noise_stddev > 0:
        layers.append(
            tf.keras.layers.GaussianNoise(stddev=config.noise_stddev)
        )

    augmentation = tf.keras.Sequential(layers, name='augmentation')

    return augmentation


class AugmentationPipeline:
    """Data augmentation pipeline for training."""

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        rotation_factor: float = 1.0,
        flip_mode: str = "horizontal_and_vertical",
        zoom_range: tuple = (-0.1, 0.1),
        brightness_factor: float = 0.2,
        contrast_factor: float = 0.2,
        noise_stddev: float = 0.01,
        enabled: bool = True
    ):
        """Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration (if provided, overrides other args)
            rotation_factor: Range for random rotation (1.0 = 360°)
            flip_mode: Flip mode ("horizontal", "vertical", "horizontal_and_vertical")
            zoom_range: Range for random zoom
            brightness_factor: Range for brightness adjustment
            contrast_factor: Range for contrast adjustment
            noise_stddev: Standard deviation for Gaussian noise
            enabled: Enable augmentation
        """
        if config is not None:
            self.config = config
        else:
            self.config = AugmentationConfig(
                enabled=enabled,
                rotation_factor=rotation_factor,
                flip_mode=flip_mode,
                zoom_range=zoom_range,
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                noise_stddev=noise_stddev
            )

        self.augmentation_layer = create_augmentation_layer(self.config)

    def __call__(self, image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Apply augmentation to image.

        Args:
            image: Input image tensor
            training: Whether in training mode (augmentation only applied during training)

        Returns:
            Augmented image tensor
        """
        if not training or not self.config.enabled:
            return image

        return self.augmentation_layer(image, training=True)

    def augment_batch(self, images: tf.Tensor) -> tf.Tensor:
        """Apply augmentation to batch of images.

        Args:
            images: Batch of images, shape (batch_size, H, W, C)

        Returns:
            Augmented batch
        """
        if not self.config.enabled:
            return images

        return self.augmentation_layer(images, training=True)


def augment_image_tf(
    image: tf.Tensor,
    label: tf.Tensor,
    augmentation_layer: tf.keras.Sequential
) -> tuple:
    """Augment image using TensorFlow operations.

    Args:
        image: Input image tensor
        label: Image label
        augmentation_layer: Augmentation layer to apply

    Returns:
        Tuple of (augmented_image, label)
    """
    augmented_image = augmentation_layer(image, training=True)
    return augmented_image, label


def get_default_augmentation() -> AugmentationPipeline:
    """Get default augmentation pipeline for bee brood detection.

    Returns:
        Configured augmentation pipeline
    """
    return AugmentationPipeline(
        rotation_factor=1.0,  # Full 360° rotation
        flip_mode="horizontal_and_vertical",
        zoom_range=(-0.1, 0.1),
        brightness_factor=0.2,
        contrast_factor=0.2,
        noise_stddev=0.01,
        enabled=True
    )
