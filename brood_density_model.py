#!/usr/bin/env python3
"""Density map model for bee brood cell counting.

Based on CSRNet architecture but simplified for brood cells.
Uses point annotations (just cell centers) to generate density maps.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter


class BroodDensityModel:
    """Density estimation model for counting brood cells."""

    def __init__(self, input_size=(512, 512, 3)):
        """Initialize density model.

        Args:
            input_size: (height, width, channels) for input images
        """
        self.input_size = input_size
        self.model = None

    def build_model(self):
        """Build CSRNet-inspired density estimation model.

        Architecture:
        - VGG-like frontend for feature extraction
        - Dilated convolutions for larger receptive field
        - Output: Density map (same size as input / 8)
        """
        inputs = keras.Input(shape=self.input_size, name='image_input')

        # Normalize input
        x = layers.Lambda(lambda x: x / 255.0 - 0.5)(inputs)

        # Frontend: VGG-like feature extraction
        # Block 1
        x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_1')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_2')(x)
        x = layers.MaxPooling2D(2, name='pool1')(x)

        # Block 2
        x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_1')(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_2')(x)
        x = layers.MaxPooling2D(2, name='pool2')(x)

        # Block 3
        x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_1')(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_2')(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_3')(x)
        x = layers.MaxPooling2D(2, name='pool3')(x)

        # Backend: Dilated convolutions for density map
        # Dilated convs maintain resolution while increasing receptive field
        x = layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name='dilated1')(x)
        x = layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name='dilated2')(x)
        x = layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name='dilated3')(x)

        x = layers.Conv2D(256, 3, padding='same', dilation_rate=2, activation='relu', name='dilated4')(x)
        x = layers.Conv2D(128, 3, padding='same', dilation_rate=2, activation='relu', name='dilated5')(x)
        x = layers.Conv2D(64, 3, padding='same', dilation_rate=2, activation='relu', name='dilated6')(x)

        # Output: Density map (1 channel)
        # Use linear activation (can have any positive value)
        density_map = layers.Conv2D(1, 1, padding='same', activation='linear', name='density_output')(x)

        self.model = keras.Model(inputs=inputs, outputs=density_map, name='brood_density_net')

        return self.model

    def compile_model(self, learning_rate=1e-4):
        """Compile model with appropriate loss and optimizer.

        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()

        # Use Mean Squared Error between predicted and ground truth density maps
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']  # Mean Absolute Error for monitoring
        )

        return self.model

    def predict_density(self, image):
        """Predict density map for an image.

        Args:
            image: Input image (H, W, 3) - BGR format from cv2

        Returns:
            density_map: Predicted density map
            count: Estimated total count (sum of density map)
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded")

        # Resize to model input size if needed
        h, w = image.shape[:2]
        target_h, target_w = self.input_size[:2]

        if h != target_h or w != target_w:
            image_resized = cv2.resize(image, (target_w, target_h))
        else:
            image_resized = image.copy()

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        image_batch = np.expand_dims(image_rgb, axis=0)

        # Predict density map
        density_map = self.model.predict(image_batch, verbose=0)[0, :, :, 0]

        # Resize density map back to original image size
        if h != target_h or w != target_w:
            # Density map is downsampled by 8x from input
            density_h, density_w = density_map.shape
            original_density_h = h // 8
            original_density_w = w // 8
            density_map = cv2.resize(density_map, (original_density_w, original_density_h))

        # Count = sum of density map
        count = np.sum(density_map)

        return density_map, count

    def visualize_density(self, image, density_map, alpha=0.6):
        """Create visualization overlay of density map on image.

        Args:
            image: Original image (H, W, 3)
            density_map: Predicted density map
            alpha: Transparency of heatmap overlay

        Returns:
            vis_img: Visualization with heatmap overlay
        """
        h, w = image.shape[:2]

        # Resize density map to match image size
        density_h, density_w = density_map.shape
        if density_h != h or density_w != w:
            density_resized = cv2.resize(density_map, (w, h))
        else:
            density_resized = density_map

        # Normalize density map to 0-255 for colormap
        density_normalized = density_resized.copy()
        density_max = density_normalized.max()
        if density_max > 0:
            density_normalized = (density_normalized / density_max * 255).astype(np.uint8)
        else:
            density_normalized = density_normalized.astype(np.uint8)

        # Apply colormap (hot = red/yellow for high density)
        heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)

        # Create mask for low-density areas (make them transparent)
        mask = density_normalized > 5  # Threshold for visibility
        mask_3d = np.stack([mask, mask, mask], axis=-1)

        # Blend heatmap with original image
        vis_img = image.copy()
        vis_img = np.where(mask_3d,
                           cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0),
                           image)

        return vis_img


def generate_density_map_from_points(points, image_shape, sigma=8):
    """Generate ground truth density map from point annotations.

    Each point represents the center of one brood cell.
    We create a Gaussian kernel at each point location.

    Args:
        points: List of (x, y) coordinates of cell centers
        image_shape: (height, width) of the image
        sigma: Standard deviation of Gaussian kernel (cell size / 3)

    Returns:
        density_map: Ground truth density map where sum = number of cells
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density_map

    # Create Gaussian kernel
    kernel_size = sigma * 3  # 3-sigma rule
    x = np.arange(-kernel_size, kernel_size + 1)
    y = np.arange(-kernel_size, kernel_size + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize so sum = 1

    # Place Gaussian at each point
    for (px, py) in points:
        # Convert to integer coordinates
        px, py = int(px), int(py)

        # Get kernel placement boundaries
        x1 = max(0, px - kernel_size)
        x2 = min(w, px + kernel_size + 1)
        y1 = max(0, py - kernel_size)
        y2 = min(h, py + kernel_size + 1)

        # Get corresponding kernel region
        kx1 = max(0, kernel_size - px)
        kx2 = kernel_size + (x2 - px)
        ky1 = max(0, kernel_size - py)
        ky2 = kernel_size + (y2 - py)

        # Add kernel to density map
        if x2 > x1 and y2 > y1:
            density_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    return density_map


def save_point_annotations(image_path, points, output_path):
    """Save point annotations to file.

    Format: One line per point
    x y

    Args:
        image_path: Path to source image (for reference)
        points: List of (x, y) tuples
        output_path: Path to save annotation file (.txt)
    """
    with open(output_path, 'w') as f:
        f.write(f"# Point annotations for {os.path.basename(image_path)}\n")
        f.write(f"# Total points: {len(points)}\n")
        for x, y in points:
            f.write(f"{x} {y}\n")


def load_point_annotations(annotation_path):
    """Load point annotations from file.

    Args:
        annotation_path: Path to annotation file (.txt)

    Returns:
        points: List of (x, y) tuples
    """
    points = []
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
    return points


if __name__ == '__main__':
    # Test model architecture
    print("Building Brood Density Model...")
    model = BroodDensityModel(input_size=(512, 512, 3))
    model.build_model()

    print(f"\nModel Summary:")
    model.model.summary()

    print(f"\nTotal parameters: {model.model.count_params():,}")

    # Test density map generation
    print("\n\nTesting density map generation...")
    test_points = [(100, 100), (150, 120), (200, 100), (250, 130)]
    density_map = generate_density_map_from_points(test_points, (512, 512), sigma=8)

    print(f"Generated density map shape: {density_map.shape}")
    print(f"Sum of density map: {density_map.sum():.2f} (should be ~{len(test_points)})")
    print(f"Max density value: {density_map.max():.4f}")

    print("\n✓ Density model architecture ready!")
