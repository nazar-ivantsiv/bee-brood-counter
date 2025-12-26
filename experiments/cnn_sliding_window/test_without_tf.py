#!/usr/bin/env python3
"""Test implementation without TensorFlow (for troubleshooting)."""
import sys
from pathlib import Path

def test_config():
    """Test configuration system."""
    print("\n" + "="*70)
    print("TEST 1: Configuration System")
    print("="*70)

    from src.config import Config

    # Test default config
    config = Config()
    print(f"✓ Default config created")
    print(f"  Model type: {config.model.model_type}")
    print(f"  Batch size: {config.training.batch_size}")

    # Test loading from YAML
    for yaml_file in ['configs/baseline.yaml', 'configs/mobilenet.yaml', 'configs/efficientnet.yaml']:
        config = Config.from_yaml(yaml_file)
        print(f"✓ Loaded {yaml_file}")
        print(f"  Model: {config.model.model_type}, Epochs: {config.training.epochs}")

    return True

def test_dataset():
    """Test dataset loading (without TensorFlow)."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Loading")
    print("="*70)

    from src.config import DataConfig
    from src.data.dataset import load_dataset_paths, split_dataset, compute_class_weights

    config = DataConfig()

    # Load paths
    image_paths, labels = load_dataset_paths(
        positive_path=config.positive_path,
        negative_path=config.negative_path,
        shuffle=False
    )

    print(f"✓ Dataset paths loaded: {len(image_paths)} total samples")
    print(f"  Positive samples: {sum(labels)}")
    print(f"  Negative samples: {len(labels) - sum(labels)}")

    # Test splitting
    splits = split_dataset(image_paths, labels, stratify=True, seed=42)
    print(f"✓ Dataset split:")
    print(f"  Train: {len(splits['train'][0])}")
    print(f"  Val: {len(splits['val'][0])}")
    print(f"  Test: {len(splits['test'][0])}")

    # Test class weights
    class_weights = compute_class_weights(labels)
    print(f"✓ Class weights computed: {class_weights}")

    return True

def test_preprocessing():
    """Test image preprocessing."""
    print("\n" + "="*70)
    print("TEST 3: Image Preprocessing")
    print("="*70)

    from src.data.preprocessing import load_image, normalize_image
    from src.config import DataConfig
    import numpy as np

    config = DataConfig()

    # Get a sample image
    positive_path = config.positive_path
    sample_images = list(positive_path.glob("*.png"))

    if len(sample_images) > 0:
        sample_image_path = sample_images[0]

        # Test loading
        img = load_image(sample_image_path, target_size=(64, 64))
        print(f"✓ Image loaded: shape {img.shape}, dtype {img.dtype}")

        # Test normalization
        normalized = normalize_image(img)
        print(f"✓ Image normalized: range [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"  Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}")

        return True
    else:
        print("✗ No sample images found")
        return False

def test_code_structure():
    """Test that all modules are importable (structure test)."""
    print("\n" + "="*70)
    print("TEST 4: Code Structure (Import Test)")
    print("="*70)

    modules = [
        ("Config", "from src.config import Config"),
        ("Preprocessing", "from src.data.preprocessing import load_image"),
        ("Dataset (no TF)", "from src.data.dataset import load_dataset_paths"),
    ]

    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            return False

    return True

def main():
    """Run non-TensorFlow tests."""
    print("\n" + "="*70)
    print("BEE BROOD COUNTER - TESTS (WITHOUT TENSORFLOW)")
    print("="*70)
    print("\nNote: Skipping TensorFlow-dependent tests due to environment issue")

    results = {}

    tests = [
        ("Code Structure", test_code_structure),
        ("Configuration", test_config),
        ("Dataset Loading", test_dataset),
        ("Preprocessing", test_preprocessing),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(results.values())
    total = len(results)

    print("\n" + "="*70)
    print(f"PASSED: {passed}/{total} tests")
    print("="*70)

    if passed == total:
        print("\n✅ All non-TensorFlow tests passed!")
        print("\n⚠️  TensorFlow tests skipped due to environment issue.")
        print("    See TENSORFLOW_FIX.md for solutions.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
