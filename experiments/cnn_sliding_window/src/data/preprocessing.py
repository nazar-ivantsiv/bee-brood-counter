"""Image preprocessing utilities for bee brood detection."""
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union


# Pixel depth for normalization
PIXEL_DEPTH = 255.0


def load_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """Load and resize image from file.

    Args:
        image_path: Path to image file
        target_size: Target size (width, height) for resizing

    Returns:
        Loaded and resized image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to target size
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def normalize_image(image: np.ndarray, pixel_depth: float = PIXEL_DEPTH) -> np.ndarray:
    """Normalize image to zero mean and unit variance.

    Args:
        image: Input image array
        pixel_depth: Maximum pixel value (default: 255.0)

    Returns:
        Normalized image with mean ≈ 0 and std ≈ 0.5
    """
    # Convert to float32 and normalize to [0, 1]
    normalized = image.astype(np.float32) / pixel_depth

    # Shift to zero mean (approximately)
    # Mean of 0.5 in [0,1] range becomes 0 after subtracting 0.5
    normalized = normalized - 0.5

    return normalized


def denormalize_image(image: np.ndarray, pixel_depth: float = PIXEL_DEPTH) -> np.ndarray:
    """Denormalize image back to [0, 255] range.

    Args:
        image: Normalized image array
        pixel_depth: Maximum pixel value (default: 255.0)

    Returns:
        Denormalized image in [0, 255] range
    """
    # Add 0.5 to shift from [-0.5, 0.5] to [0, 1]
    denormalized = (image + 0.5) * pixel_depth

    # Clip and convert to uint8
    denormalized = np.clip(denormalized, 0, pixel_depth).astype(np.uint8)

    return denormalized


def histogram_normalization(
    image: np.ndarray,
    clahe: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply histogram normalization to improve contrast.

    Args:
        image: Input image (RGB)
        clahe: Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clip_limit: Threshold for contrast limiting (for CLAHE)
        tile_grid_size: Size of grid for histogram equalization (for CLAHE)

    Returns:
        Image with normalized histogram
    """
    # Convert to LAB color space for better histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    if clahe:
        # Apply CLAHE to L channel
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        l = clahe_obj.apply(l)
    else:
        # Standard histogram equalization
        l = cv2.equalizeHist(l)

    # Merge channels back
    lab = cv2.merge([l, a, b])

    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


def apply_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    sigma: float = 0.0
) -> np.ndarray:
    """Apply Gaussian blur to image.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Gaussian kernel standard deviation

    Returns:
        Blurred image
    """
    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        raise ValueError("Kernel size must be positive")

    return cv2.GaussianBlur(image, kernel_size, sigma)


def make_feature_vec(
    image: np.ndarray,
    target_size: Tuple[int, int, int] = (64, 64, 3),
    normalize: bool = True,
    hist_norm: bool = False,
    blur_kernel: Tuple[int, int] = None
) -> np.ndarray:
    """Convert image to feature vector for model input.

    Args:
        image: Input image
        target_size: Target shape (height, width, channels)
        normalize: Apply normalization
        hist_norm: Apply histogram normalization
        blur_kernel: Apply Gaussian blur with specified kernel size

    Returns:
        Preprocessed image ready for model input, shape (1, H, W, C)
    """
    # Ensure correct size
    if image.shape[:2] != target_size[:2]:
        image = cv2.resize(image, target_size[:2], interpolation=cv2.INTER_LINEAR)

    # Apply histogram normalization if requested
    if hist_norm:
        image = histogram_normalization(image)

    # Apply blur if requested
    if blur_kernel is not None:
        image = apply_blur(image, blur_kernel)

    # Normalize pixel values
    if normalize:
        image = normalize_image(image)

    # Add batch dimension
    feature_vec = np.expand_dims(image, axis=0)

    return feature_vec.astype(np.float32)


def preprocess_dataset_image(image_path: Union[str, Path]) -> tf.Tensor:
    """Preprocess image for tf.data pipeline.

    Args:
        image_path: Path to image file (as string or tensor)

    Returns:
        Preprocessed image tensor
    """
    # Read image file (image_path is already a tensor in tf.data pipeline)
    img = tf.io.read_file(image_path)

    # Decode image (PNG format)
    img = tf.image.decode_png(img, channels=3)

    # Resize to 64x64
    img = tf.image.resize(img, [64, 64], method='bilinear')

    # Normalize to [-0.5, 0.5]
    img = tf.cast(img, tf.float32) / 255.0 - 0.5

    return img


def random_augment_image(
    image: tf.Tensor,
    rotation_factor: float = 1.0,
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    brightness_range: float = 0.2,
    contrast_range: float = 0.2,
    zoom_range: float = 0.1,
    seed: int = None
) -> tf.Tensor:
    """Apply random augmentations to image.

    Args:
        image: Input image tensor
        rotation_factor: Range for random rotation (1.0 = 360°)
        flip_horizontal: Enable horizontal flipping
        flip_vertical: Enable vertical flipping
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        zoom_range: Range for zoom
        seed: Random seed for reproducibility

    Returns:
        Augmented image tensor
    """
    # Random rotation
    if rotation_factor > 0:
        # TensorFlow rotation expects radians
        max_delta = rotation_factor * 2 * np.pi
        angle = tf.random.uniform([], -max_delta, max_delta, seed=seed)
        image = tf.keras.ops.image.affine_transform(
            image,
            transform=[
                tf.cos(angle), -tf.sin(angle), 0,
                tf.sin(angle), tf.cos(angle), 0,
                0, 0
            ]
        )

    # Random horizontal flip
    if flip_horizontal:
        image = tf.image.random_flip_left_right(image, seed=seed)

    # Random vertical flip
    if flip_vertical:
        image = tf.image.random_flip_up_down(image, seed=seed)

    # Random brightness
    if brightness_range > 0:
        image = tf.image.random_brightness(image, brightness_range, seed=seed)

    # Random contrast
    if contrast_range > 0:
        lower = 1.0 - contrast_range
        upper = 1.0 + contrast_range
        image = tf.image.random_contrast(image, lower, upper, seed=seed)

    # Ensure values stay in valid range
    image = tf.clip_by_value(image, -0.5, 0.5)

    return image


class ImagePreprocessor:
    """Reusable image preprocessor for consistent preprocessing."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        hist_norm: bool = False,
        blur_kernel: Tuple[int, int] = None
    ):
        """Initialize preprocessor.

        Args:
            target_size: Target image size (width, height)
            normalize: Apply normalization
            hist_norm: Apply histogram normalization
            blur_kernel: Apply Gaussian blur with specified kernel size
        """
        self.target_size = target_size
        self.normalize = normalize
        self.hist_norm = hist_norm
        self.blur_kernel = blur_kernel

    def __call__(self, image_path: Union[str, Path]) -> np.ndarray:
        """Preprocess image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image
        """
        # Load image
        img = load_image(image_path, self.target_size)

        # Apply histogram normalization
        if self.hist_norm:
            img = histogram_normalization(img)

        # Apply blur
        if self.blur_kernel is not None:
            img = apply_blur(img, self.blur_kernel)

        # Normalize
        if self.normalize:
            img = normalize_image(img)

        return img
