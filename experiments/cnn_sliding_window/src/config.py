"""Configuration management for bee brood counter training pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = "./dataset"
    positive_subdir: str = "positive"
    negative_subdir: str = "negative"
    img_size: Tuple[int, int, int] = (64, 64, 3)
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42

    @property
    def positive_path(self) -> Path:
        """Get full path to positive samples."""
        return Path(self.dataset_path) / self.positive_subdir

    @property
    def negative_path(self) -> Path:
        """Get full path to negative samples."""
        return Path(self.dataset_path) / self.negative_subdir


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = True
    rotation_factor: float = 1.0  # Full 360° rotation
    flip_mode: str = "horizontal_and_vertical"
    zoom_range: Tuple[float, float] = (-0.1, 0.1)
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    noise_stddev: float = 0.01


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "baseline"  # baseline, mobilenet_v2, efficientnet_b0
    input_shape: Tuple[int, int, int] = (64, 64, 3)
    num_classes: int = 2

    # Transfer learning specific
    freeze_base: bool = True  # For first stage training
    base_trainable: bool = False  # Start with frozen base

    # Baseline CNN specific
    conv1_filters: int = 16
    conv2_filters: int = 16
    fc_units: int = 64

    # Transfer model classifier head
    global_pooling: str = "avg"  # avg or max
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.2])


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.005
    optimizer: str = "adam"  # adam, rmsprop, sgd

    # Learning rate schedule
    lr_decay_enabled: bool = True
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 100

    # Class imbalance handling
    use_class_weights: bool = True
    class_weight_ratio: float = 4.3  # negative:positive ratio

    # Regularization
    l2_reg: float = 0.0
    label_smoothing: float = 0.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_f1_score"
    early_stopping_mode: str = "max"

    # Model checkpoint
    save_best_only: bool = True
    checkpoint_monitor: str = "val_f1_score"
    checkpoint_mode: str = "max"

    # Two-stage training (for transfer learning)
    stage1_epochs: int = 15
    stage1_lr: float = 1e-3
    stage2_epochs: int = 35
    stage2_lr: float = 1e-5


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration."""
    enabled: bool = False
    n_splits: int = 5
    shuffle: bool = True
    stratified: bool = True


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    tracking_uri: str = "./experiments"
    experiment_name: str = "bee_brood_counter"
    run_name: Optional[str] = None
    log_models: bool = True
    log_artifacts: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Global settings
    gpu_id: int = 0
    mixed_precision: bool = False
    verbose: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object with loaded settings
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            cross_validation=CrossValidationConfig(**config_dict.get('cross_validation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            **{k: v for k, v in config_dict.items()
               if k not in ['data', 'augmentation', 'model', 'training', 'cross_validation', 'experiment']}
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where to save YAML configuration
        """
        config_dict = {
            'data': self.data.__dict__,
            'augmentation': self.augmentation.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'cross_validation': self.cross_validation.__dict__,
            'experiment': self.experiment.__dict__,
            'gpu_id': self.gpu_id,
            'mixed_precision': self.mixed_precision,
            'verbose': self.verbose,
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            'data': self.data.__dict__,
            'augmentation': self.augmentation.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'cross_validation': self.cross_validation.__dict__,
            'experiment': self.experiment.__dict__,
            'gpu_id': self.gpu_id,
            'mixed_precision': self.mixed_precision,
            'verbose': self.verbose,
        }


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Config object with default settings
    """
    return Config()
