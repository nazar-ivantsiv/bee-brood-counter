#!/usr/bin/env python3
"""Demo script showing integrated brood cell detection with BeeFrame."""
import os
import sys
import argparse
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bee_frame import BeeFrame


def main():
    """Run brood cell detection demo."""
    parser = argparse.ArgumentParser(
        description="Detect brood cells in bee frame images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect brood cells with default settings
  python demo_detection.py --image /path/to/frame.png

  # Adjust detection thresholds
  python demo_detection.py --image /path/to/frame.png --threshold 0.6 --confidence 0.8

  # Save result to file
  python demo_detection.py --image /path/to/frame.png --output result.png

  # CPU-only mode (for stability on M1 Macs)
  python demo_detection.py --image /path/to/frame.png --cpu-only
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='./best_baseline_model.keras',
        help='Path to model file (default: ./best_baseline_model.keras)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to bee frame image'
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
        help='Minimum confidence to keep detection (default: 0.7)'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.3,
        help='NMS IoU threshold (default: 0.3)'
    )
    parser.add_argument(
        '--no-nms',
        action='store_true',
        help='Disable non-maximum suppression'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save detection result to file'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display result (useful for batch processing)'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only inference'
    )

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu_only:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only")

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print(f"Please ensure the model file exists at: {args.model}")
        return 1

    print("=" * 80)
    print("BEE BROOD CELL DETECTION")
    print("=" * 80)

    # Create BeeFrame detector with model
    print(f"\nInitializing detector...")
    frame = BeeFrame(model_path=args.model)

    # Load image
    image_dir = os.path.dirname(args.image)
    image_name = os.path.basename(args.image)

    print(f"\nLoading image: {args.image}")
    frame.load_image(image_dir, image_name)

    # Run detection
    detections = frame.detect_brood_cells(
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        use_nms=not args.no_nms,
        nms_threshold=args.nms_threshold
    )

    # Get summary
    summary = frame.get_detection_summary()

    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total brood cells detected: {summary['total_detections']}")

    if summary['total_detections'] > 0:
        print(f"Average confidence:        {summary['avg_confidence']:.4f}")
        print(f"Confidence range:          {summary['min_confidence']:.4f} - {summary['max_confidence']:.4f}")
        print(f"Confidence std dev:        {summary['std_confidence']:.4f}")

    # Visualize
    print("\nGenerating visualization...")
    vis_img = frame.visualize_detections(show_confidence=True, min_confidence=0.0)

    # Save if requested
    if args.output:
        cv2.imwrite(args.output, vis_img)
        print(f"✓ Result saved to: {args.output}")

    # Display if not disabled
    if not args.no_display:
        print("\nDisplaying result...")
        print("Press 'q' to quit, 's' to save")

        cv2.namedWindow('Brood Cell Detection', cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow('Brood Cell Detection', vis_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save with default name
                output_name = f"detection_result_{os.path.basename(args.image)}"
                cv2.imwrite(output_name, vis_img)
                print(f"✓ Saved to: {output_name}")

        cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print("DETECTION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
