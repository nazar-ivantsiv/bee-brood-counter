#!/usr/bin/env python3
"""
Comprehensive test script for bee brood counter implementation.

Run this after installing dependencies:
    pip install -r requirements.txt
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)

    tests_passed = 0
    tests_total = 0

    modules = [
        ("Config", "from src.config import Config"),
        ("Preprocessing", "from src.data.preprocessing import load_image, normalize_image"),
        ("Augmentation", "from src.data.augmentation import AugmentationPipeline"),
        ("Dataset", "from src.data.dataset import DatasetLoader"),
        ("Baseline CNN", "from src.models.baseline_cnn import create_baseline_cnn"),
        ("Transfer Models", "from src.models.transfer_models import create_mobilenet_v2"),
        ("Model Builder", "from src.models.model_builder import build_model"),
    ]

    for name, import_stmt in modules:
        tests_total += 1
        try:
            exec(import_stmt)
            print(f"✓ {name}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_config():
    """Test configuration loading."""
    print("\n" + "="*70)
    print("TEST 2: Configuration Loading")
    print("="*70)

    from src.config import Config

    tests_passed = 0
    tests_total = 0

    # Test default config
    tests_total += 1
    try:
        config = Config()
        assert config.model.model_type == "baseline"
        assert config.training.batch_size == 16
        print("✓ Default configuration")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Default configuration: {e}")

    # Test loading from YAML
    for config_file in ['configs/baseline.yaml', 'configs/mobilenet.yaml', 'configs/efficientnet.yaml']:
        tests_total += 1
        try:
            config = Config.from_yaml(config_file)
            assert config.model.model_type is not None
            assert config.training.batch_size > 0
            print(f"✓ Load from {config_file}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ Load from {config_file}: {e}")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_dataset_paths():
    """Test dataset paths and structure."""
    print("\n" + "="*70)
    print("TEST 3: Dataset Paths")
    print("="*70)

    tests_passed = 0
    tests_total = 0

    # Check dataset directory exists
    tests_total += 1
    dataset_path = Path("./dataset")
    if dataset_path.exists():
        print(f"✓ Dataset directory exists: {dataset_path}")
        tests_passed += 1
    else:
        print(f"✗ Dataset directory not found: {dataset_path}")

    # Check positive samples
    tests_total += 1
    positive_path = dataset_path / "positive"
    if positive_path.exists():
        num_positive = len(list(positive_path.glob("*.png")))
        print(f"✓ Positive samples directory: {num_positive} images")
        tests_passed += 1
    else:
        print(f"✗ Positive samples directory not found")

    # Check negative samples
    tests_total += 1
    negative_path = dataset_path / "negative"
    if negative_path.exists():
        num_negative = len(list(negative_path.glob("*.png")))
        print(f"✓ Negative samples directory: {num_negative} images")
        tests_passed += 1
    else:
        print(f"✗ Negative samples directory not found")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_data_pipeline():
    """Test data loading and preprocessing."""
    print("\n" + "="*70)
    print("TEST 4: Data Pipeline")
    print("="*70)

    import numpy as np
    from src.config import Config, DataConfig, AugmentationConfig
    from src.data.dataset import load_dataset_paths, split_dataset, compute_class_weights
    from src.data.preprocessing import load_image, normalize_image

    tests_passed = 0
    tests_total = 0

    config = DataConfig()

    # Test loading dataset paths
    tests_total += 1
    try:
        image_paths, labels = load_dataset_paths(
            positive_path=config.positive_path,
            negative_path=config.negative_path,
            shuffle=False
        )
        assert len(image_paths) > 0
        assert len(image_paths) == len(labels)
        print(f"✓ Load dataset paths: {len(image_paths)} total samples")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Load dataset paths: {e}")
        return False

    # Test dataset splitting
    tests_total += 1
    try:
        splits = split_dataset(image_paths, labels, stratify=True)
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        train_size = len(splits['train'][0])
        val_size = len(splits['val'][0])
        test_size = len(splits['test'][0])
        print(f"✓ Dataset splitting: train={train_size}, val={val_size}, test={test_size}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Dataset splitting: {e}")

    # Test class weights computation
    tests_total += 1
    try:
        class_weights = compute_class_weights(labels)
        assert 0 in class_weights
        assert 1 in class_weights
        print(f"✓ Class weights: {class_weights}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Class weights: {e}")

    # Test image loading
    tests_total += 1
    try:
        if len(image_paths) > 0:
            img = load_image(image_paths[0], target_size=(64, 64))
            assert img.shape == (64, 64, 3)
            print(f"✓ Image loading: shape {img.shape}")
            tests_passed += 1
    except Exception as e:
        print(f"✗ Image loading: {e}")

    # Test normalization
    tests_total += 1
    try:
        if len(image_paths) > 0:
            img = load_image(image_paths[0], target_size=(64, 64))
            normalized = normalize_image(img)
            assert normalized.dtype == np.float32
            assert normalized.min() >= -0.5
            assert normalized.max() <= 0.5
            print(f"✓ Image normalization: range [{normalized.min():.2f}, {normalized.max():.2f}]")
            tests_passed += 1
    except Exception as e:
        print(f"✗ Image normalization: {e}")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_models():
    """Test model creation."""
    print("\n" + "="*70)
    print("TEST 5: Model Creation")
    print("="*70)

    import numpy as np
    from src.models.baseline_cnn import create_baseline_cnn
    from src.models.transfer_models import create_mobilenet_v2, create_efficientnet_b0
    from src.models.model_builder import build_model
    from src.config import Config

    tests_passed = 0
    tests_total = 0

    # Test baseline CNN
    tests_total += 1
    try:
        model = create_baseline_cnn()
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 2)

        # Test forward pass
        test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)
        output = model(test_input)
        assert output.shape == (1, 2)

        print(f"✓ Baseline CNN: {model.count_params():,} parameters")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Baseline CNN: {e}")

    # Test MobileNetV2
    tests_total += 1
    try:
        model = create_mobilenet_v2(freeze_base=True)
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 2)

        # Test forward pass
        test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)
        output = model(test_input)
        assert output.shape == (1, 2)

        print(f"✓ MobileNetV2: {model.count_params():,} parameters")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MobileNetV2: {e}")

    # Test EfficientNetB0
    tests_total += 1
    try:
        model = create_efficientnet_b0(freeze_base=True)
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 2)

        # Test forward pass
        test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)
        output = model(test_input)
        assert output.shape == (1, 2)

        print(f"✓ EfficientNetB0: {model.count_params():,} parameters")
        tests_passed += 1
    except Exception as e:
        print(f"✗ EfficientNetB0: {e}")

    # Test model builder
    tests_total += 1
    try:
        config = Config.from_yaml('configs/baseline.yaml')
        model = build_model(config.model, config.training)
        assert model is not None
        print(f"✓ Model builder (baseline): {model.name}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Model builder: {e}")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_augmentation():
    """Test data augmentation."""
    print("\n" + "="*70)
    print("TEST 6: Data Augmentation")
    print("="*70)

    import numpy as np
    import tensorflow as tf
    from src.data.augmentation import AugmentationPipeline
    from src.config import AugmentationConfig

    tests_passed = 0
    tests_total = 0

    # Test augmentation pipeline creation
    tests_total += 1
    try:
        config = AugmentationConfig(enabled=True)
        aug_pipeline = AugmentationPipeline(config=config)
        print("✓ Augmentation pipeline created")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Augmentation pipeline: {e}")
        return False

    # Test augmentation on image
    tests_total += 1
    try:
        test_image = tf.random.normal((64, 64, 3))
        augmented = aug_pipeline(test_image, training=True)
        assert augmented.shape == (64, 64, 3)
        print("✓ Augmentation applied successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Augmentation application: {e}")

    # Test augmentation disabled
    tests_total += 1
    try:
        config_no_aug = AugmentationConfig(enabled=False)
        aug_pipeline_disabled = AugmentationPipeline(config=config_no_aug)
        result = aug_pipeline_disabled(test_image, training=True)
        assert result.shape == test_image.shape
        print("✓ Augmentation disabled mode works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Augmentation disabled: {e}")

    print(f"\nPassed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BEE BROOD COUNTER - IMPLEMENTATION TEST SUITE")
    print("="*70)

    all_tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Paths", test_dataset_paths),
        ("Data Pipeline", test_data_pipeline),
        ("Model Creation", test_models),
        ("Augmentation", test_augmentation),
    ]

    results = {}

    for test_name, test_func in all_tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} test suites passed")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! Implementation is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
