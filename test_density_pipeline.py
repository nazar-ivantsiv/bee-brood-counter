#!/usr/bin/env python3
"""
Integration tests for density map pipeline.
Verifies all components are functional after cleanup.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Test results
tests_passed = 0
tests_failed = 0
test_messages = []


def test_result(test_name, passed, message=""):
    """Record test result."""
    global tests_passed, tests_failed, test_messages

    if passed:
        tests_passed += 1
        status = "✓ PASS"
    else:
        tests_failed += 1
        status = "✗ FAIL"

    output = f"{status}: {test_name}"
    if message:
        output += f" - {message}"
    test_messages.append(output)
    print(output)


def test_imports():
    """Test 1: Verify all required imports work."""
    print("\n=== Test 1: Import Dependencies ===")

    try:
        import tensorflow as tf
        test_result("TensorFlow import", True, f"v{tf.__version__}")
    except ImportError as e:
        test_result("TensorFlow import", False, str(e))
        return False

    try:
        import cv2
        test_result("OpenCV import", True, f"v{cv2.__version__}")
    except ImportError as e:
        test_result("OpenCV import", False, str(e))
        return False

    try:
        import numpy as np
        test_result("NumPy import", True, f"v{np.__version__}")
    except ImportError as e:
        test_result("NumPy import", False, str(e))
        return False

    try:
        import scipy
        from scipy.ndimage import gaussian_filter
        test_result("SciPy import", True, f"v{scipy.__version__}")
    except ImportError as e:
        test_result("SciPy import", False, str(e))
        return False

    return True


def test_core_modules():
    """Test 2: Verify core density map modules can be imported."""
    print("\n=== Test 2: Core Module Imports ===")

    try:
        from brood_density_model import BroodDensityModel
        test_result("BroodDensityModel import", True)
    except ImportError as e:
        test_result("BroodDensityModel import", False, str(e))
        return False

    # Check if annotation script exists
    if os.path.exists("annotate_cell_centers.py"):
        test_result("annotate_cell_centers.py exists", True)
    else:
        test_result("annotate_cell_centers.py exists", False)
        return False

    # Check if training script exists
    if os.path.exists("train_density_model.py"):
        test_result("train_density_model.py exists", True)
    else:
        test_result("train_density_model.py exists", False)
        return False

    # Check if inference script exists
    if os.path.exists("run_density_detection.py"):
        test_result("run_density_detection.py exists", True)
    else:
        test_result("run_density_detection.py exists", False)
        return False

    return True


def test_model_architecture():
    """Test 3: Verify model architecture can be built."""
    print("\n=== Test 3: Model Architecture ===")

    try:
        from brood_density_model import BroodDensityModel

        model = BroodDensityModel(input_size=(512, 512, 3))
        test_result("Model initialization", True)

        # Build model
        model.build_model()
        test_result("Model build", True)

        # Check model structure
        if model.model is not None:
            test_result("Model created", True)

            # Verify input shape
            input_shape = model.model.input_shape
            if input_shape == (None, 512, 512, 3):
                test_result("Input shape correct", True, f"{input_shape}")
            else:
                test_result("Input shape correct", False, f"Expected (None, 512, 512, 3), got {input_shape}")

            # Verify output shape (should be 1/8 of input)
            output_shape = model.model.output_shape
            if output_shape == (None, 64, 64, 1):
                test_result("Output shape correct", True, f"{output_shape}")
            else:
                test_result("Output shape correct", False, f"Expected (None, 64, 64, 1), got {output_shape}")
        else:
            test_result("Model created", False)
            return False

    except Exception as e:
        test_result("Model architecture", False, str(e))
        return False

    return True


def test_density_map_generation():
    """Test 4: Verify density prediction works."""
    print("\n=== Test 4: Density Prediction ===")

    try:
        from brood_density_model import BroodDensityModel
        import tensorflow as tf

        model = BroodDensityModel(input_size=(512, 512, 3))
        model.build_model()
        test_result("Model built for density prediction", True)

        # Create dummy image
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Predict density (returns tuple: density_map, count)
        density_map, count = model.predict_density(dummy_img)
        test_result("Density prediction executed", True)

        # Verify density map properties
        if density_map.shape == (64, 64):  # 1/8 of input
            test_result("Density map shape", True, f"{density_map.shape}")
        else:
            test_result("Density map shape", False, f"Expected (64, 64), got {density_map.shape}")

        # Verify output is numeric
        if isinstance(density_map, np.ndarray):
            test_result("Density map is numpy array", True)
        else:
            test_result("Density map is numpy array", False)

        # Verify count is returned
        if isinstance(count, (int, float, np.number)):
            test_result("Count returned", True, f"{count:.1f}")
        else:
            test_result("Count returned", False)

    except Exception as e:
        test_result("Density prediction", False, str(e))
        return False

    return True


def test_sample_image_exists():
    """Test 5: Verify sample image and annotations exist."""
    print("\n=== Test 5: Sample Data ===")

    sample_image = "annotated_frames/bee_frame_sample.png"
    sample_annotations = "annotated_frames/bee_frame_sample_annotations.txt"

    if os.path.exists(sample_image):
        test_result("bee_frame_sample.png exists", True, sample_image)

        # Try to load it
        img = cv2.imread(sample_image)
        if img is not None:
            test_result("Sample image readable", True, f"shape={img.shape}")
        else:
            test_result("Sample image readable", False)
    else:
        test_result("bee_frame_sample.png exists", False)

    if os.path.exists(sample_annotations):
        test_result("bee_frame_sample_annotations.txt exists", True, sample_annotations)

        # Try to load annotations
        try:
            points = []
            with open(sample_annotations, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments and empty lines
                        parts = line.split()  # Split by whitespace
                        if len(parts) == 2:
                            x, y = map(int, parts)
                            points.append((x, y))
            test_result("Sample annotations readable", True, f"{len(points)} points")
        except Exception as e:
            test_result("Sample annotations readable", False, str(e))
    else:
        test_result("bee_frame_sample_annotations.txt exists", False)


def test_model_file_exists():
    """Test 6: Verify trained model exists."""
    print("\n=== Test 6: Trained Model ===")

    # Check for model in models folder
    model_path = "models/best_density_model.keras"
    if os.path.exists(model_path):
        test_result("best_density_model.keras exists", True, model_path)

        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        test_result("Model file size", True, f"{size_mb:.1f} MB")

        # Try to load model
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, safe_mode=False)
            test_result("Model loading", True)

            # Verify model can make predictions
            dummy_input = np.zeros((1, 512, 512, 3), dtype=np.float32)
            output = model.predict(dummy_input, verbose=0)
            test_result("Model inference", True, f"output shape={output.shape}")

        except Exception as e:
            test_result("Model loading", False, str(e))
    else:
        test_result("best_density_model.keras exists", False, "Model not found in models/")


def test_inference_pipeline():
    """Test 7: Run full inference pipeline on sample image."""
    print("\n=== Test 7: Full Inference Pipeline ===")

    sample_image = "annotated_frames/bee_frame_sample.png"
    model_path = "models/best_density_model.keras"

    if not os.path.exists(sample_image):
        test_result("Inference pipeline", False, f"Sample image not found at {sample_image}")
        return False

    if not os.path.exists(model_path):
        test_result("Inference pipeline", False, f"Model not found at {model_path}")
        return False

    try:
        from brood_density_model import BroodDensityModel
        import tensorflow as tf

        # Load model
        model = BroodDensityModel(input_size=(512, 512, 3))
        model.model = tf.keras.models.load_model(model_path, safe_mode=False)
        test_result("Model loaded for inference", True)

        # Load image
        img = cv2.imread(sample_image)
        test_result("Sample image loaded", True, f"shape={img.shape}")

        # Run prediction (returns tuple: density_map, count)
        density_map, predicted_count = model.predict_density(img)
        test_result("Prediction executed", True)

        # Verify results
        if predicted_count > 0:
            test_result("Predicted count", True, f"{predicted_count:.0f} cells")
        else:
            test_result("Predicted count", False, "Count is 0 or negative")

        if density_map is not None and density_map.shape[0] > 0:
            test_result("Density map generated", True, f"shape={density_map.shape}")
        else:
            test_result("Density map generated", False)

        # Verify count is reasonable (should be ~2500 based on previous results)
        if 2000 <= predicted_count <= 3000:
            test_result("Count within expected range", True, f"{predicted_count:.0f} cells")
        else:
            test_result("Count within expected range", False, f"{predicted_count:.0f} cells (expected ~2500)")

    except Exception as e:
        test_result("Inference pipeline", False, str(e))
        return False

    return True


def test_no_cnn_dependencies():
    """Test 8: Verify CNN-specific files are not in main folder."""
    print("\n=== Test 8: CNN Files Removed ===")

    # Files that should be moved to experiments/
    cnn_files = [
        "bee_frame.py",
        "run_brood_counter.py",
        "demo_detection.py",
        "test_integration.py",
        "best_baseline_model.keras",
        "nms.py"
    ]

    all_removed = True
    for file in cnn_files:
        if os.path.exists(file):
            test_result(f"{file} not in main folder", False, "Should be in experiments/")
            all_removed = False
        else:
            test_result(f"{file} not in main folder", True)

    # Check they exist in experiments/
    if os.path.exists("experiments/cnn_sliding_window/"):
        test_result("experiments/cnn_sliding_window/ exists", True)

        # Check a few key files are there
        if os.path.exists("experiments/cnn_sliding_window/bee_frame.py"):
            test_result("CNN files in experiments/", True)
        else:
            test_result("CNN files in experiments/", False, "bee_frame.py not found")
    else:
        test_result("experiments/cnn_sliding_window/ exists", False)

    return all_removed


def test_requirements():
    """Test 9: Verify requirements.txt has minimal dependencies."""
    print("\n=== Test 9: Requirements File ===")

    if not os.path.exists("requirements.txt"):
        test_result("requirements.txt exists", False)
        return False

    test_result("requirements.txt exists", True)

    # Read requirements
    with open("requirements.txt", 'r') as f:
        content = f.read()

    # Check for core dependencies
    required_deps = ["tensorflow", "opencv-python", "numpy", "scipy"]
    for dep in required_deps:
        if dep in content:
            test_result(f"{dep} in requirements.txt", True)
        else:
            test_result(f"{dep} in requirements.txt", False)

    # Check that CNN-specific deps are NOT present
    unwanted_deps = ["mlflow", "keras-tuner", "pytest", "black", "flake8", "mypy", "scikit-learn"]
    for dep in unwanted_deps:
        if dep not in content:
            test_result(f"{dep} not in requirements.txt", True)
        else:
            test_result(f"{dep} not in requirements.txt", False, "Should be in experiments/requirements_cnn.txt")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DENSITY MAP PIPELINE INTEGRATION TESTS")
    print("=" * 60)

    # Run all tests
    test_imports()
    test_core_modules()
    test_model_architecture()
    test_density_map_generation()
    test_sample_image_exists()
    test_model_file_exists()
    test_inference_pipeline()
    test_no_cnn_dependencies()
    test_requirements()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total Tests:  {tests_passed + tests_failed}")
    print(f"Success Rate: {100 * tests_passed / (tests_passed + tests_failed):.1f}%")
    print("=" * 60)

    if tests_failed > 0:
        print("\n⚠ SOME TESTS FAILED - Review output above")
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED - Density map pipeline is fully functional!")
        sys.exit(0)


if __name__ == "__main__":
    main()
