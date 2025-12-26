#!/usr/bin/env python3
"""Test the integrated model with sample images."""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from bee_frame import BeeFrame


def test_single_cell_classification():
    """Test classification of single 60x60 cell images."""
    print("=" * 80)
    print("TESTING MODEL INTEGRATION - SINGLE CELL CLASSIFICATION")
    print("=" * 80)

    # Initialize detector
    print("\n1. Loading model...")
    frame = BeeFrame(model_path='./best_baseline_model.keras')

    # Test positive samples
    print("\n2. Testing positive samples (brood cells)...")
    positive_samples = [
        'dataset/positive/102.png',
        'dataset/positive/1133.png',
        'dataset/positive/1135.png'
    ]

    positive_results = []
    for img_path in positive_samples:
        if not os.path.exists(img_path):
            print(f"  ⚠️  Skipping {img_path} (not found)")
            continue

        # Load and preprocess
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Could not load {img_path}")
            continue

        preprocessed = frame.preprocess_for_model(img)

        # Predict
        prediction = frame.model.predict(preprocessed, verbose=0)
        confidence = prediction[0][1]  # Positive class probability

        positive_results.append(confidence)
        status = "✓" if confidence > 0.5 else "✗"
        print(f"  {status} {os.path.basename(img_path)}: {confidence:.4f} (brood)")

    # Test negative samples
    print("\n3. Testing negative samples (non-brood cells)...")
    negative_samples = [
        'dataset/negative/11.png',
        'dataset/negative/13.png',
        'dataset/negative/14.png'
    ]

    negative_results = []
    for img_path in negative_samples:
        if not os.path.exists(img_path):
            print(f"  ⚠️  Skipping {img_path} (not found)")
            continue

        # Load and preprocess
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Could not load {img_path}")
            continue

        preprocessed = frame.preprocess_for_model(img)

        # Predict
        prediction = frame.model.predict(preprocessed, verbose=0)
        confidence = prediction[0][1]  # Positive class probability

        negative_results.append(confidence)
        status = "✓" if confidence < 0.5 else "✗"
        print(f"  {status} {os.path.basename(img_path)}: {confidence:.4f} (non-brood)")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if positive_results:
        correct_positive = sum(1 for c in positive_results if c > 0.5)
        print(f"Positive samples: {correct_positive}/{len(positive_results)} correct")
        print(f"  Average confidence: {np.mean(positive_results):.4f}")

    if negative_results:
        correct_negative = sum(1 for c in negative_results if c < 0.5)
        print(f"Negative samples: {correct_negative}/{len(negative_results)} correct")
        print(f"  Average confidence: {np.mean(negative_results):.4f}")

    total_correct = sum(1 for c in positive_results if c > 0.5) + sum(1 for c in negative_results if c < 0.5)
    total_samples = len(positive_results) + len(negative_results)

    if total_samples > 0:
        accuracy = total_correct / total_samples * 100
        print(f"\nOverall accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")

        if accuracy >= 90:
            print("✓ Integration test PASSED!")
        else:
            print("⚠️  Lower than expected accuracy")
    else:
        print("\n⚠️  No samples found to test")

    return accuracy >= 90 if total_samples > 0 else False


if __name__ == '__main__':
    try:
        success = test_single_cell_classification()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
