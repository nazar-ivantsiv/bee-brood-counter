"""Dataset loading and preparation for bee brood detection."""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold

from ..config import DataConfig, AugmentationConfig
from .preprocessing import preprocess_dataset_image
from .augmentation import create_augmentation_layer


def load_dataset_paths(
    positive_path: Path,
    negative_path: Path,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """Load image paths and labels from dataset directories.

    Args:
        positive_path: Path to positive samples directory
        negative_path: Path to negative samples directory
        shuffle: Shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []

    # Load positive samples (label = 1)
    if not positive_path.exists():
        raise FileNotFoundError(f"Positive samples path not found: {positive_path}")

    positive_files = sorted(positive_path.glob("*.png"))
    image_paths.extend([str(f) for f in positive_files])
    labels.extend([1] * len(positive_files))

    # Load negative samples (label = 0)
    if not negative_path.exists():
        raise FileNotFoundError(f"Negative samples path not found: {negative_path}")

    negative_files = sorted(negative_path.glob("*.png"))
    image_paths.extend([str(f) for f in negative_files])
    labels.extend([0] * len(negative_files))

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle if requested
    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(image_paths))
        image_paths = image_paths[indices]
        labels = labels[indices]

    return image_paths.tolist(), labels.tolist()


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    stratify: bool = True,
    seed: int = 42
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Split dataset into train, validation, and test sets.

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        stratify: Use stratified splitting to maintain class balance
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (paths, labels)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    stratify_labels = labels if stratify else None

    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_split,
        stratify=stratify_labels,
        random_state=seed
    )

    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    stratify_train_val = train_val_labels if stratify else None

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio,
        stratify=stratify_train_val,
        random_state=seed
    )

    return {
        'train': (train_paths.tolist(), train_labels.tolist()),
        'val': (val_paths.tolist(), val_labels.tolist()),
        'test': (test_paths.tolist(), test_labels.tolist())
    }


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute class weights to handle imbalanced dataset.

    Args:
        labels: List of labels

    Returns:
        Dictionary mapping class indices to weights
    """
    labels = np.array(labels)
    unique_classes = np.unique(labels)

    # Compute weights inversely proportional to class frequencies
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = {}
    for cls in unique_classes:
        # Weight = total_samples / (num_classes * class_count)
        weight = total_samples / (len(unique_classes) * class_counts[cls])
        class_weights[int(cls)] = float(weight)

    return class_weights


def create_tf_dataset(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    augmentation_config: Optional[AugmentationConfig] = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    prefetch_size: int = tf.data.AUTOTUNE,
    cache: bool = False,
    repeat: bool = False
) -> tf.data.Dataset:
    """Create TensorFlow dataset from image paths.

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size
        augmentation_config: Augmentation configuration (None = no augmentation)
        shuffle: Shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        prefetch_size: Prefetch buffer size
        cache: Cache dataset in memory
        repeat: Repeat dataset indefinitely

    Returns:
        TensorFlow dataset
    """
    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Map preprocessing function
    def load_and_preprocess(path, label):
        image = preprocess_dataset_image(path)
        # Convert label to one-hot encoding for categorical crossentropy
        label_onehot = tf.one_hot(label, depth=2)
        return image, label_onehot

    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache if requested (after loading, before augmentation)
    if cache:
        dataset = dataset.cache()

    # Apply augmentation if provided
    if augmentation_config is not None and augmentation_config.enabled:
        augmentation_layer = create_augmentation_layer(augmentation_config)

        def augment(image, label):
            augmented_image = augmentation_layer(image, training=True)
            return augmented_image, label

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    dataset = dataset.batch(batch_size)

    # Repeat if requested
    if repeat:
        dataset = dataset.repeat()

    # Prefetch
    dataset = dataset.prefetch(buffer_size=prefetch_size)

    return dataset


def create_datasets_from_config(
    config: DataConfig,
    augmentation_config: AugmentationConfig,
    batch_size: int = 32,
    cache: bool = False
) -> Dict[str, tf.data.Dataset]:
    """Create train, validation, and test datasets from configuration.

    Args:
        config: Data configuration
        augmentation_config: Augmentation configuration
        batch_size: Batch size
        cache: Cache datasets in memory

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    # Load all image paths and labels
    image_paths, labels = load_dataset_paths(
        positive_path=config.positive_path,
        negative_path=config.negative_path,
        shuffle=True,
        seed=config.seed
    )

    # Split dataset
    splits = split_dataset(
        image_paths=image_paths,
        labels=labels,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        stratify=True,
        seed=config.seed
    )

    # Create TF datasets
    datasets = {}

    # Training dataset (with augmentation)
    train_paths, train_labels = splits['train']
    datasets['train'] = create_tf_dataset(
        image_paths=train_paths,
        labels=train_labels,
        batch_size=batch_size,
        augmentation_config=augmentation_config,
        shuffle=True,
        cache=cache
    )

    # Validation dataset (no augmentation)
    val_paths, val_labels = splits['val']
    datasets['val'] = create_tf_dataset(
        image_paths=val_paths,
        labels=val_labels,
        batch_size=batch_size,
        augmentation_config=None,  # No augmentation for validation
        shuffle=False,
        cache=cache
    )

    # Test dataset (no augmentation)
    test_paths, test_labels = splits['test']
    datasets['test'] = create_tf_dataset(
        image_paths=test_paths,
        labels=test_labels,
        batch_size=batch_size,
        augmentation_config=None,  # No augmentation for test
        shuffle=False,
        cache=cache
    )

    # Compute class weights from training labels
    datasets['class_weights'] = compute_class_weights(train_labels)

    # Store dataset sizes
    datasets['sizes'] = {
        'train': len(train_labels),
        'val': len(val_labels),
        'test': len(test_labels)
    }

    return datasets


def create_cross_validation_splits(
    image_paths: List[str],
    labels: List[int],
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Tuple[List[str], List[int]]]]:
    """Create k-fold cross-validation splits.

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        n_splits: Number of folds
        shuffle: Shuffle before splitting
        seed: Random seed

    Returns:
        List of dictionaries, each containing 'train' and 'val' splits
    """
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    splits = []
    for train_idx, val_idx in skf.split(image_paths, labels):
        train_paths = image_paths[train_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_paths = image_paths[val_idx].tolist()
        val_labels = labels[val_idx].tolist()

        splits.append({
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels)
        })

    return splits


class DatasetLoader:
    """Dataset loader with caching and easy access."""

    def __init__(self, data_config: DataConfig, batch_size: int = 32):
        """Initialize dataset loader.

        Args:
            data_config: Data configuration
            batch_size: Batch size for datasets
        """
        self.config = data_config
        self.batch_size = batch_size
        self._image_paths = None
        self._labels = None
        self._splits = None

    def load_data(self) -> Tuple[List[str], List[int]]:
        """Load all data paths and labels.

        Returns:
            Tuple of (image_paths, labels)
        """
        if self._image_paths is None or self._labels is None:
            self._image_paths, self._labels = load_dataset_paths(
                positive_path=self.config.positive_path,
                negative_path=self.config.negative_path,
                shuffle=True,
                seed=self.config.seed
            )

        return self._image_paths, self._labels

    def get_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """Get train/val/test splits.

        Returns:
            Dictionary with splits
        """
        if self._splits is None:
            image_paths, labels = self.load_data()
            self._splits = split_dataset(
                image_paths=image_paths,
                labels=labels,
                train_split=self.config.train_split,
                val_split=self.config.val_split,
                test_split=self.config.test_split,
                stratify=True,
                seed=self.config.seed
            )

        return self._splits

    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for training data.

        Returns:
            Dictionary mapping class indices to weights
        """
        splits = self.get_splits()
        _, train_labels = splits['train']
        return compute_class_weights(train_labels)

    def get_dataset_info(self) -> Dict[str, any]:
        """Get dataset information.

        Returns:
            Dictionary with dataset statistics
        """
        image_paths, labels = self.load_data()
        splits = self.get_splits()

        labels_array = np.array(labels)
        unique, counts = np.unique(labels_array, return_counts=True)

        return {
            'total_samples': len(labels),
            'positive_samples': int(counts[1]) if len(counts) > 1 else 0,
            'negative_samples': int(counts[0]),
            'class_balance_ratio': float(counts[0] / counts[1]) if len(counts) > 1 else 0,
            'train_size': len(splits['train'][1]),
            'val_size': len(splits['val'][1]),
            'test_size': len(splits['test'][1]),
            'img_size': self.config.img_size,
        }
