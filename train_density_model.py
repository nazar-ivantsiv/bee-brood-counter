#!/usr/bin/env python3
"""Train density estimation model for brood cell counting.

Usage:
    python train_density_model.py --data-dir annotated_frames/
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from brood_density_model import (
    BroodDensityModel,
    load_point_annotations,
    generate_density_map_from_points
)


class DensityDataGenerator(keras.utils.Sequence):
    """Data generator for training density estimation model."""

    def __init__(self, image_paths, annotation_paths, batch_size=4,
                 input_size=(512, 512), sigma=8, shuffle=True):
        """Initialize data generator.

        Args:
            image_paths: List of paths to training images
            annotation_paths: List of paths to corresponding annotation files
            batch_size: Number of samples per batch
            input_size: (height, width) to resize images
            sigma: Gaussian sigma for density map generation
            shuffle: Whether to shuffle data each epoch
        """
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.sigma = sigma
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))

        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        # Generate data for this batch
        images = []
        density_maps = []

        for idx in batch_indexes:
            image_path = self.image_paths[idx]
            annotation_path = self.annotation_paths[idx]

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load {image_path}, skipping")
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to input size
            h, w = image.shape[:2]
            target_h, target_w = self.input_size
            image_resized = cv2.resize(image, (target_w, target_h))

            # Load point annotations
            points = load_point_annotations(annotation_path)

            # Scale points to match resized image
            scale_x = target_w / w
            scale_y = target_h / h
            points_scaled = [(x * scale_x, y * scale_y) for x, y in points]

            # Generate density map at 1/8 resolution (to match model output)
            density_h = target_h // 8
            density_w = target_w // 8

            # Scale points to density map size
            scale_x = density_w / target_w
            scale_y = density_h / target_h
            points_density = [(x * scale_x, y * scale_y) for x, y in points_scaled]

            # Generate ground truth density map
            density_map = generate_density_map_from_points(
                points_density,
                (density_h, density_w),
                sigma=self.sigma
            )

            images.append(image_resized)
            density_maps.append(density_map)

        # Convert to arrays
        images = np.array(images, dtype=np.float32)
        density_maps = np.array(density_maps, dtype=np.float32)
        density_maps = np.expand_dims(density_maps, axis=-1)  # Add channel dimension

        return images, density_maps

    def on_epoch_end(self):
        """Shuffle indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def find_image_annotation_pairs(data_dir):
    """Find all image-annotation pairs in a directory.

    Args:
        data_dir: Directory containing images and annotation files

    Returns:
        image_paths: List of image file paths
        annotation_paths: List of corresponding annotation file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    annotation_suffix = '_annotations.txt'

    image_paths = []
    annotation_paths = []

    # Find all images
    all_files = os.listdir(data_dir)

    for filename in all_files:
        # Check if it's an image file
        name, ext = os.path.splitext(filename)
        if ext.lower() not in image_extensions:
            continue

        image_path = os.path.join(data_dir, filename)

        # Look for corresponding annotation file
        annotation_filename = name + annotation_suffix
        annotation_path = os.path.join(data_dir, annotation_filename)

        if os.path.exists(annotation_path):
            image_paths.append(image_path)
            annotation_paths.append(annotation_path)
        else:
            print(f"Warning: No annotation file for {filename}, skipping")

    return image_paths, annotation_paths


def train_model(data_dir, epochs=100, batch_size=4, val_split=0.2,
                input_size=(512, 512), learning_rate=1e-4):
    """Train density estimation model.

    Args:
        data_dir: Directory with annotated images
        epochs: Number of training epochs
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        input_size: (height, width) for model input
        learning_rate: Learning rate for optimizer

    Returns:
        model: Trained model
        history: Training history
    """
    # Find all image-annotation pairs
    print("Finding annotated images...")
    image_paths, annotation_paths = find_image_annotation_pairs(data_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No annotated images found in {data_dir}")

    print(f"Found {len(image_paths)} annotated images")

    # Split into train/val
    n_samples = len(image_paths)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_images = [image_paths[i] for i in train_indices]
    train_annotations = [annotation_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_annotations = [annotation_paths[i] for i in val_indices]

    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")

    # Create data generators
    train_generator = DensityDataGenerator(
        train_images,
        train_annotations,
        batch_size=batch_size,
        input_size=input_size,
        sigma=8,
        shuffle=True
    )

    val_generator = DensityDataGenerator(
        val_images,
        val_annotations,
        batch_size=batch_size,
        input_size=input_size,
        sigma=8,
        shuffle=False
    ) if n_val > 0 else None

    # Build and compile model
    print("\nBuilding density model...")
    density_model = BroodDensityModel(input_size=(*input_size, 3))
    model = density_model.compile_model(learning_rate=learning_rate)

    print(f"Total parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_density_model.keras',
            save_best_only=True,
            monitor='val_loss' if val_generator else 'loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_generator else 'loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_generator else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    print("\nStarting training...")
    print("=" * 80)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model saved to: best_density_model.keras")

    # Show final metrics
    if val_generator:
        final_val_loss = min(history.history['val_loss'])
        final_val_mae = min(history.history['val_mae'])
        print(f"Best validation loss: {final_val_loss:.4f}")
        print(f"Best validation MAE: {final_val_mae:.2f} cells")

    return model, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train density estimation model for brood counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on annotated images in a directory
  python train_density_model.py --data-dir annotated_frames/

  # Train with custom parameters
  python train_density_model.py --data-dir annotated_frames/ --epochs 200 --batch-size 8
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing annotated images and annotation files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training (default: 4)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split fraction (default: 0.2)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only training'
    )

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu_only:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only")

    # Check data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    # Train model
    try:
        model, history = train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            learning_rate=args.learning_rate
        )
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
