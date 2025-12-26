#!/usr/bin/env python3
"""Interactive tool for annotating brood cell centers with point clicks.

Much faster than drawing bounding boxes - just click the center of each cell!

Usage:
    python annotate_cell_centers.py --image bee_frame.jpg

Controls:
    Left Click - Add point at cell center
    Right Click - Remove nearest point
    'z' - Undo last point
    's' - Save annotations
    'q' - Quit without saving
    'c' - Clear all points
    '+' / '=' - Increase point size
    '-' / '_' - Decrease point size
    'h' - Toggle heatmap preview
"""

import os
import sys
import argparse
import cv2
import numpy as np
from brood_density_model import (
    save_point_annotations,
    load_point_annotations,
    generate_density_map_from_points
)


class CellCenterAnnotator:
    """Interactive annotator for marking brood cell centers."""

    def __init__(self, image_path, annotation_path=None):
        """Initialize annotator.

        Args:
            image_path: Path to bee frame image
            annotation_path: Path to save/load annotations (optional)
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.h, self.w = self.image.shape[:2]
        self.display_image = self.image.copy()

        # Annotation file
        if annotation_path is None:
            base_name = os.path.splitext(image_path)[0]
            self.annotation_path = base_name + '_annotations.txt'
        else:
            self.annotation_path = annotation_path

        # Load existing annotations if available
        if os.path.exists(self.annotation_path):
            print(f"Loading existing annotations from: {self.annotation_path}")
            self.points = load_point_annotations(self.annotation_path)
            print(f"Loaded {len(self.points)} points")
        else:
            self.points = []

        # UI state
        self.point_radius = 5
        self.point_color = (0, 255, 0)  # Green
        self.hover_color = (0, 255, 255)  # Yellow
        self.show_heatmap = False
        self.window_name = f'Annotate Cell Centers - {os.path.basename(image_path)}'

    def draw_points(self):
        """Draw all annotated points on the image."""
        self.display_image = self.image.copy()

        # Draw density heatmap if enabled
        if self.show_heatmap and len(self.points) > 0:
            density_map = generate_density_map_from_points(
                self.points,
                (self.h, self.w),
                sigma=8
            )

            # Normalize and apply colormap
            density_normalized = density_map.copy()
            if density_normalized.max() > 0:
                density_normalized = (density_normalized / density_normalized.max() * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)

            # Blend with original image
            mask = density_normalized > 5
            mask_3d = np.stack([mask, mask, mask], axis=-1)
            self.display_image = np.where(
                mask_3d,
                cv2.addWeighted(self.image, 0.5, heatmap, 0.5, 0),
                self.image
            )

        # Draw each point
        for i, (x, y) in enumerate(self.points):
            # Draw cross at point center
            cv2.drawMarker(
                self.display_image,
                (int(x), int(y)),
                self.point_color,
                cv2.MARKER_CROSS,
                self.point_radius * 2,
                1
            )
            # Draw small circle
            cv2.circle(
                self.display_image,
                (int(x), int(y)),
                self.point_radius,
                self.point_color,
                1
            )

        # Draw count overlay
        count_text = f"Points: {len(self.points)}"
        cv2.putText(
            self.display_image,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Draw instructions
        instructions = [
            "Left Click: Add | Right Click: Remove | 'z': Undo",
            "'s': Save | 'q': Quit | 'h': Toggle heatmap"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_image,
                text,
                (10, self.h - 40 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    def find_nearest_point(self, x, y, max_distance=15):
        """Find nearest point to (x, y).

        Args:
            x, y: Query coordinates
            max_distance: Maximum distance to consider

        Returns:
            index of nearest point, or None if no point within max_distance
        """
        if len(self.points) == 0:
            return None

        points_array = np.array(self.points)
        distances = np.sqrt((points_array[:, 0] - x)**2 + (points_array[:, 1] - y)**2)
        nearest_idx = np.argmin(distances)

        if distances[nearest_idx] <= max_distance:
            return nearest_idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append((x, y))
            self.draw_points()
            cv2.imshow(self.window_name, self.display_image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest point
            nearest_idx = self.find_nearest_point(x, y)
            if nearest_idx is not None:
                del self.points[nearest_idx]
                self.draw_points()
                cv2.imshow(self.window_name, self.display_image)

    def run(self):
        """Run the annotation tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.draw_points()
        cv2.imshow(self.window_name, self.display_image)

        print("\n" + "=" * 80)
        print("BROOD CELL CENTER ANNOTATION TOOL")
        print("=" * 80)
        print(f"Image: {self.image_path}")
        print(f"Annotations: {self.annotation_path}")
        print(f"Current points: {len(self.points)}")
        print("\nControls:")
        print("  Left Click     - Add point at cell center")
        print("  Right Click    - Remove nearest point")
        print("  'z'            - Undo last point")
        print("  's'            - Save annotations")
        print("  'q' / ESC      - Quit")
        print("  'c'            - Clear all points")
        print("  'h'            - Toggle density heatmap preview")
        print("  '+' / '-'      - Adjust point size")
        print("=" * 80 + "\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quitting without saving...")
                break

            elif key == ord('s'):
                # Save annotations
                save_point_annotations(self.image_path, self.points, self.annotation_path)
                print(f"✓ Saved {len(self.points)} points to: {self.annotation_path}")

            elif key == ord('z'):
                # Undo last point
                if len(self.points) > 0:
                    self.points.pop()
                    self.draw_points()
                    cv2.imshow(self.window_name, self.display_image)
                    print(f"Undo - Points: {len(self.points)}")

            elif key == ord('c'):
                # Clear all points
                if len(self.points) > 0:
                    confirm = input(f"Clear all {len(self.points)} points? (y/n): ")
                    if confirm.lower() == 'y':
                        self.points = []
                        self.draw_points()
                        cv2.imshow(self.window_name, self.display_image)
                        print("Cleared all points")

            elif key == ord('h'):
                # Toggle heatmap
                self.show_heatmap = not self.show_heatmap
                self.draw_points()
                cv2.imshow(self.window_name, self.display_image)
                print(f"Heatmap preview: {'ON' if self.show_heatmap else 'OFF'}")

            elif key in [ord('+'), ord('=')]:
                # Increase point size
                self.point_radius = min(20, self.point_radius + 1)
                self.draw_points()
                cv2.imshow(self.window_name, self.display_image)

            elif key in [ord('-'), ord('_')]:
                # Decrease point size
                self.point_radius = max(3, self.point_radius - 1)
                self.draw_points()
                cv2.imshow(self.window_name, self.display_image)

        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Annotate brood cell centers by clicking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a single image
  python annotate_cell_centers.py --image bee_frame.jpg

  # Specify custom annotation file location
  python annotate_cell_centers.py --image bee_frame.jpg --output annotations.txt
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to bee frame image to annotate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save annotations (default: <image>_annotations.txt)'
    )

    args = parser.parse_args()

    # Check image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    # Run annotator
    annotator = CellCenterAnnotator(args.image, args.output)
    annotator.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
