#!/usr/bin/env python3
"""Run brood cell detection on bee frame images."""
import os
import sys
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from bee_frame import BeeFrame


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Detect brood cells in bee frame images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default sample image
  python run_brood_counter.py

  # Specify your own image
  python run_brood_counter.py --image /path/to/your/frame.png

  # Adjust detection thresholds
  python run_brood_counter.py --image frame.png --threshold 0.6

  # CPU-only mode (for 8GB M1 Macs)
  python run_brood_counter.py --cpu-only

  # Save result without displaying
  python run_brood_counter.py --image frame.png --output result.png --no-display
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        default='bee_frame_sample.png',
        help='Path to bee frame image (default: bee_frame_sample.png)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./best_baseline_model.keras',
        help='Path to model file (default: ./best_baseline_model.keras)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Minimum confidence threshold (default: 0.7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='brood_detection_result.png',
        help='Output image path (default: brood_detection_result.png)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display result window'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only mode (recommended for 8GB M1 Macs)'
    )
    parser.add_argument(
        '--cell-size',
        type=int,
        default=None,
        help='Cell size in pixels (default: auto-detect or 60). Use smaller values (6-15) for full frames, 60 for cropped cells.'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=None,
        help='Step size for sliding window (default: cell_size // 2)'
    )
    parser.add_argument(
        '--auto-calibrate',
        action='store_true',
        help='Automatically estimate optimal cell size before detection'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Manually calibrate cell size using interactive tool'
    )

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu_only:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only for stability")

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        print("\nAvailable sample images:")
        for img in ['bee_frame_sample.png', 'DSC_0139.jpg']:
            if os.path.exists(img):
                print(f"  - {img}")
        print("\nUsage: python run_brood_counter.py --image <your_image>")
        return 1

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please ensure the model file exists.")
        print("Run the hyperparameter tuning first or use the provided model.")
        return 1

    print("=" * 80)
    print("BEE BROOD CELL COUNTER")
    print("=" * 80)
    print(f"\nImage: {args.image}")
    print(f"Model: {args.model}")
    print(f"Thresholds: classification={args.threshold}, confidence={args.confidence}")

    # Determine cell size and step size
    cell_size = args.cell_size
    step_size = args.step_size

    # Get image path components
    image_dir = os.path.dirname(args.image) if os.path.dirname(args.image) else '.'
    image_name = os.path.basename(args.image)

    # Manual calibration mode (interactive)
    if args.calibrate:
        print("\n" + "-" * 80)
        print("MANUAL CALIBRATION MODE")
        print("-" * 80)
        print("Instructions:")
        print("  1. Click and drag to draw a box around ONE brood cell")
        print("  2. Press 's' to save the calibration")
        print("  3. Press 'q' to cancel")
        print("-" * 80)

        # Load image for calibration
        temp_frame = BeeFrame()
        temp_frame.load_image(image_dir, image_name)

        # Run calibration tool
        temp_frame.get_cell_size()

        # Get calibrated size
        cell_size = temp_frame.cell_size
        step_size = temp_frame.step_size
        print(f"\n✓ Calibrated cell size: {cell_size}px")
        print(f"  Step size: {step_size}px")

    # Auto-calibration mode (automatic estimation)
    elif args.auto_calibrate and cell_size is None:
        print("\n" + "-" * 80)
        print("AUTO-CALIBRATION MODE")
        print("-" * 80)
        print("Estimating optimal cell size...")

        # Initialize temporary frame with model
        temp_frame = BeeFrame(model_path=args.model)
        temp_frame.load_image(image_dir, image_name)

        # Estimate cell size
        cell_size = temp_frame.estimate_cell_size(verbose=True)
        print(f"  Step size will be: {cell_size // 2}px")

    # Initialize BeeFrame with model and correct cell size
    print("\n" + "-" * 80)
    print("Initializing detector...")
    print("-" * 80)
    frame = BeeFrame(
        model_path=args.model,
        cell_size=cell_size if cell_size is not None else 60,
        step_size=step_size
    )

    # Display detection parameters
    print(f"Detection parameters:")
    print(f"  Cell size: {frame.cell_size}px")
    print(f"  Step size: {frame.step_size}px")
    print(f"  Approx. cells per 100x100px area: ~{(100 // frame.cell_size) ** 2}")

    # Load image (skip if already loaded during calibration)
    if not hasattr(frame, 'image'):
        print("\nLoading image...")
        frame.load_image(path=image_dir, file_name=image_name)
        print(f"✓ Image loaded: {frame.image.width}x{frame.image.height} pixels")
    else:
        print(f"\n✓ Image already loaded: {frame.image.width}x{frame.image.height} pixels")

    # Detect brood cells
    print("\n" + "-" * 80)
    print("Running detection...")
    print("-" * 80)
    detections = frame.detect_brood_cells(
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        use_nms=True,
        nms_threshold=0.3
    )

    # Get statistics
    summary = frame.get_detection_summary()

    # Print results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    print(f"\n🐝 Brood cells detected: {summary['total_detections']}")

    if summary['total_detections'] > 0:
        print(f"\nConfidence Statistics:")
        print(f"  Average:  {summary['avg_confidence']:.2%}")
        print(f"  Range:    {summary['min_confidence']:.2%} - {summary['max_confidence']:.2%}")
        print(f"  Std Dev:  {summary['std_confidence']:.4f}")

        # Estimate brood area (rough approximation)
        cell_area = frame.cell_size * frame.cell_size
        total_brood_area = summary['total_detections'] * cell_area
        frame_area = frame.image.width * frame.image.height
        coverage = (total_brood_area / frame_area) * 100
        print(f"\nEstimated brood coverage: {coverage:.1f}% of frame")
    else:
        print("\nNo brood cells detected.")
        print("Try lowering the thresholds with --threshold and --confidence")

    # Visualize results
    print("\n" + "-" * 80)
    print("Generating visualization...")
    print("-" * 80)
    result_img = frame.visualize_detections(show_confidence=True, min_confidence=0.0)

    # Save result
    cv2.imwrite(args.output, result_img)
    print(f"✓ Result saved to: {args.output}")

    # Display if not disabled
    if not args.no_display:
        print("\nDisplaying result...")
        print("  Press 'q' to quit")
        print("  Press 's' to save a copy")

        cv2.namedWindow('Brood Cell Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Brood Cell Detection', 1200, 900)

        while True:
            cv2.imshow('Brood Cell Detection', result_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_name = f"saved_{args.output}"
                cv2.imwrite(save_name, result_img)
                print(f"  ✓ Saved copy to: {save_name}")

        cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print("DETECTION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Detected {summary['total_detections']} brood cells")
    print(f"  • Result saved to {args.output}")
    print(f"  • Average confidence: {summary.get('avg_confidence', 0):.2%}")

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