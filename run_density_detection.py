#!/usr/bin/env python3
"""Run density estimation inference on bee frame images.

Usage:
    python run_density_detection.py --image bee_frame.jpg --model best_density_model.keras
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from brood_density_model import BroodDensityModel


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run density estimation on bee frame images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run detection on single image
  python run_density_detection.py --image bee_frame.jpg --model best_density_model.keras

  # Save output
  python run_density_detection.py --image bee_frame.jpg --model best_density_model.keras --output result.jpg
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to bee frame image'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained density model (.keras file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='density_detection_result.png',
        help='Output image path (default: density_detection_result.png)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display result window'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only mode'
    )

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu_only:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only")

    # Check files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return 1

    print("=" * 80)
    print("BEE BROOD DENSITY ESTIMATION")
    print("=" * 80)
    print(f"\nImage: {args.image}")
    print(f"Model: {args.model}")

    # Load image
    print("\nLoading image...")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return 1

    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h} pixels")

    # Load model
    print("\nLoading density model...")
    density_model = BroodDensityModel(input_size=(512, 512, 3))
    density_model.model = tf.keras.models.load_model(args.model, safe_mode=False)
    print(f"✓ Model loaded")
    print(f"  Parameters: {density_model.model.count_params():,}")

    # Run inference
    print("\n" + "-" * 80)
    print("Running density estimation...")
    print("-" * 80)

    import time
    start_time = time.time()

    density_map, predicted_count = density_model.predict_density(image)

    inference_time = time.time() - start_time

    print(f"✓ Inference complete ({inference_time:.2f} seconds)")

    # Print results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    print(f"\n🐝 Estimated brood cells: {predicted_count:.0f}")

    # Density map statistics
    print(f"\nDensity Map Statistics:")
    print(f"  Shape: {density_map.shape}")
    print(f"  Max density: {density_map.max():.4f}")
    print(f"  Min density: {density_map.min():.4f}")
    print(f"  Mean density: {density_map.mean():.4f}")

    # Visualize results
    print("\n" + "-" * 80)
    print("Generating visualization...")
    print("-" * 80)

    vis_img = density_model.visualize_density(image, density_map, alpha=0.6)

    # Add count overlay
    cv2.putText(
        vis_img,
        f"Count: {predicted_count:.0f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    # Save result
    cv2.imwrite(args.output, vis_img)
    print(f"✓ Result saved to: {args.output}")

    # Display if not disabled
    if not args.no_display:
        print("\nDisplaying result...")
        print("  Press any key to close")

        cv2.namedWindow('Density Estimation Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Density Estimation Result', 1200, 900)
        cv2.imshow('Density Estimation Result', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print("DETECTION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Estimated {predicted_count:.0f} brood cells")
    print(f"  • Inference time: {inference_time:.2f} seconds")
    print(f"  • Result saved to: {args.output}")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
