import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Model-based detection disabled.")

class BeeFrame(object):
    """"""
    WIN_NAME = 'Bee Frame'
    STEP_TO_CELL_SIZE_RATIO = 4  # 1/4
    
    class Image(object):
        """Image container."""

        NUM_CHANNELS = 3 # RGB
        IMG_SIZE = (64, 64) # Size of image
        PIXEL_DEPTH = 255.0  # Number of levels per pixel.
        
        def __init__(self, path, file_name):
            img_path = os.path.join(path, file_name)
            if self.NUM_CHANNELS == 3:
                self._img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                self._img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.height, self.width = self._img.shape[:2]
            self._img = self._img.copy()

        def add_weighted(self, mask, mask_weight=0.7):
            self._img = cv2.addWeighted(src1=self._img, \
                                        alpha=1 - mask_weight, \
                                        src2=mask, \
                                        beta=mask_weight, \
                                        gamma=0)

        def apply_mask(self, mask):
            self._img = cv2.bitwise_and(self._img, self._img, mask=mask)

        def blur(self, kernel = (3, 3)):
            """Smooth (blur) image."""
            self._img = cv2.blur(self._img, kernel)
            return self._img

        def hitogram_normalization(self, clahe=False):
            """Image histogram normalization.
            Args:
                clahe -- use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            """
            if clahe:
                # create a CLAHE object (Arguments are optional).
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                self._img = clahe.apply(self._img)
            else:
                img_hist_equalized = cv2.cvtColor(self._img, cv2.COLOR_BGR2YCrCb)
                img_hist_equalized[:, :, 0] = cv2.equalizeHist(img_hist_equalized[:, :, 0])
                self._img = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
            return self._img

        def draw_circle(self, coords, img=np.array([])):
            x, y, end_x, end_y = coords
            if not len(img):
                img = self._img
            width = end_x - x
            height = end_y - y
            cv2.ellipse(img=img, \
                        center=(x + width//2, y + height//2), \
                        axes=(width//2, height//2),
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=(0,255,0),
                        thickness=2)

        def draw_rect(self, coords, img=np.array([])):
            x, y, end_x, end_y = coords
            if not len(img):
                img = self._img
            cv2.rectangle(img=img,
                          pt1=(x, y),
                          pt2=(end_x, end_y),
                          color=(0,255,0),
                          thickness=1)

    
    def __init__(self, model_path=None, cell_size=60, step_size=None):
        """Initialize BeeFrame detector.

        Args:
            model_path: Path to trained .keras model file (optional)
            cell_size: Size of detection window in pixels (default: 60)
            step_size: Sliding window step size in pixels (default: cell_size // 2)
        """
        self.cell_size = cell_size
        self.step_size = step_size if step_size is not None else cell_size // 2
        self.start_x = 0
        self.start_y = 0
        self.model = None
        self.detections = []

        # Load model if path provided
        if model_path is not None:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load a trained model for brood detection.

        Args:
            model_path: Path to saved .keras model file
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot load model.")

        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded: {self.model.name}")
        print(f"  Parameters: {self.model.count_params():,}")

    def load_image(self, path, file_name):
        if hasattr(self, 'image'):
            del self.image
        self.image = self.Image(path, file_name)
        # Clear previous detections
        self.detections = []

    def get_cell_size(self):
        """Tool to manually measure bee frame cell size.
        Press:
            'q' -- Quit
            's' -- Save acquired parameters
        """

        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)  
        data = {}
        data['drawing'] = False # true if mouse is pressed
        data['ix_iy'] = -1, -1
        data['tmp_img'] = self.image._img.copy()
        data['original_img'] = self.image._img.copy()

        def mouse_handler(event, x, y, flags, data):
            """Mouse callback function."""
            if event == cv2.EVENT_LBUTTONDOWN:
                data['drawing'] = True
                data['ix_iy'] = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if data['drawing'] == True:
                    cv2.rectangle(data['tmp_img'],data['ix_iy'],(x,y),(0,255,0), 2)
                    #data['tmp_img'] = data['original_img']
            elif event == cv2.EVENT_LBUTTONUP:
                data['drawing'] = False
                cv2.rectangle(data['tmp_img'],data['ix_iy'],(x,y),(0,255,0), 2)
                data['x_y'] = x, y

        cv2.setMouseCallback(self.WIN_NAME, mouse_handler, data)
        while True:
            cv2.imshow(self.WIN_NAME, data['tmp_img'])
            data['tmp_img'] = self.image._img.copy()
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == ord('q'):
                # Quit
                break
            elif key_pressed ==ord('s'):
                # Save
                self.cell_size = max(abs(data['x_y'][0] - data['ix_iy'][0]),\
                                    abs(data['x_y'][1] - data['ix_iy'][1]))
                self.step_size = self.cell_size // self.STEP_TO_CELL_SIZE_RATIO
                self.start_x, self.start_y = data['ix_iy']
                break
        cv2.destroyAllWindows()


    
    def sliding_window(self):
        """Slide a window across the image."""
        for y in range(0, self.image.height, self.step_size):
            for x in range(0, self.image.width, self.step_size):
                # yield the current window
                window = self.image._img[y:y + self.cell_size, \
                                    x:x + self.cell_size]
                if window.shape[0] == window.shape[1]:
                    yield (x, y, window)

    def preprocess_for_model(self, window):
        """Preprocess a window for model input.

        Args:
            window: 60x60 or 64x64 BGR image from sliding window

        Returns:
            Preprocessed image ready for model (1, 64, 64, 3) normalized to [-0.5, 0.5]
        """
        # Convert BGR to RGB
        window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)

        # Resize to 64x64 if needed
        if window_rgb.shape[:2] != (64, 64):
            window_rgb = cv2.resize(window_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Normalize to [-0.5, 0.5] (same as training)
        normalized = window_rgb.astype(np.float32) / 255.0 - 0.5

        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)

        return batch

    def detect_brood_cells(self, threshold=0.5, confidence_threshold=0.7,
                          use_nms=True, nms_threshold=0.3):
        """Detect brood cells in the loaded image using the model.

        Args:
            threshold: Classification threshold (default: 0.5)
            confidence_threshold: Minimum confidence to keep detection (default: 0.7)
            use_nms: Apply non-maximum suppression to remove overlapping detections
            nms_threshold: IoU threshold for NMS (default: 0.3)

        Returns:
            List of detections: [(x, y, end_x, end_y, confidence), ...]
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Use load_model() first.")

        if not hasattr(self, 'image'):
            raise RuntimeError("No image loaded. Use load_image() first.")

        print(f"\nDetecting brood cells...")
        print(f"  Image size: {self.image.width}x{self.image.height}")
        print(f"  Cell size: {self.cell_size}x{self.cell_size}")
        print(f"  Step size: {self.step_size}")
        print(f"  Classification threshold: {threshold}")
        print(f"  Confidence threshold: {confidence_threshold}")

        detections = []
        total_windows = 0

        # Slide window across image
        for x, y, window in self.sliding_window():
            total_windows += 1

            # Preprocess window
            preprocessed = self.preprocess_for_model(window)

            # Predict
            prediction = self.model.predict(preprocessed, verbose=0)
            confidence = float(prediction[0][1])  # Probability of positive class

            # Check if brood cell detected
            if confidence >= threshold:
                detections.append({
                    'bbox': (x, y, x + self.cell_size, y + self.cell_size),
                    'confidence': confidence
                })

        print(f"  Total windows: {total_windows}")
        print(f"  Raw detections: {len(detections)}")

        # Filter by confidence threshold
        detections = [d for d in detections if d['confidence'] >= confidence_threshold]
        print(f"  After confidence filtering: {len(detections)}")

        # Apply non-maximum suppression if requested
        if use_nms and len(detections) > 0:
            detections = self._apply_nms(detections, nms_threshold)
            print(f"  After NMS: {len(detections)}")

        # Store detections
        self.detections = detections

        print(f"✓ Detection complete: {len(detections)} brood cells found")

        return detections

    def _apply_nms(self, detections, iou_threshold=0.3):
        """Apply non-maximum suppression to remove overlapping detections.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            iou_threshold: IoU threshold for considering boxes as overlapping

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # Calculate areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence (descending)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            # Keep highest confidence box
            i = order[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep only boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def visualize_detections(self, show_confidence=True, min_confidence=0.0):
        """Visualize detected brood cells on the image.

        Args:
            show_confidence: Show confidence scores on detections
            min_confidence: Minimum confidence to display (for filtering visualization)

        Returns:
            Image with detections drawn
        """
        if not hasattr(self, 'image'):
            raise RuntimeError("No image loaded.")

        # Create a copy for visualization
        vis_img = self.image._img.copy()

        # Filter detections by confidence if needed
        detections_to_show = [d for d in self.detections
                             if d['confidence'] >= min_confidence]

        # Draw each detection
        for detection in detections_to_show:
            x, y, end_x, end_y = detection['bbox']
            confidence = detection['confidence']

            # Color based on confidence (green = high, yellow = medium, red = low)
            if confidence >= 0.9:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.75:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (end_x, end_y), color, 2)

            # Draw confidence score if requested
            if show_confidence:
                label = f"{confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1

                # Get text size for background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle
                cv2.rectangle(
                    vis_img,
                    (x, y - text_height - 4),
                    (x + text_width, y),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    vis_img,
                    label,
                    (x, y - 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )

        # Draw count
        count_text = f"Brood cells: {len(detections_to_show)}"
        cv2.putText(
            vis_img,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        return vis_img

    def get_detection_summary(self):
        """Get summary statistics of detections.

        Returns:
            Dictionary with detection statistics
        """
        if len(self.detections) == 0:
            return {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }

        confidences = [d['confidence'] for d in self.detections]

        return {
            'total_detections': len(self.detections),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'std_confidence': np.std(confidences)
        }

    def estimate_cell_size(self, verbose=True):
        """Estimate appropriate cell size by testing multiple scales.

        This method runs quick detections at different cell sizes and returns
        the size that produces a reasonable number of detections (not too few,
        not too many).

        Args:
            verbose: Print progress information (default: True)

        Returns:
            Estimated cell size in pixels
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Use load_model() first.")

        if not hasattr(self, 'image'):
            raise RuntimeError("No image loaded. Use load_image() first.")

        if verbose:
            print("Testing different cell sizes...")

        # Store original values
        original_cell_size = self.cell_size
        original_step_size = self.step_size

        best_size = 60  # Default fallback
        best_score = float('inf')

        # Test cell sizes from 6px to 80px
        test_sizes = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]

        for test_size in test_sizes:
            # Update cell size temporarily
            self.cell_size = test_size
            self.step_size = max(test_size // 3, 2)  # Smaller step for better coverage

            try:
                # Run quick detection with low thresholds
                detections = self.detect_brood_cells(
                    threshold=0.3,
                    confidence_threshold=0.5,
                    use_nms=True,
                    nms_threshold=0.4
                )

                num_detections = len(detections)

                if verbose:
                    print(f"  {test_size}px: {num_detections} detections", end="")

                # Score based on how close to ideal range (30-150 detections)
                # This range works well for typical bee frames
                ideal_min, ideal_max = 30, 150

                if num_detections < ideal_min:
                    score = (ideal_min - num_detections) * 2  # Penalize too few
                elif num_detections > ideal_max:
                    score = (num_detections - ideal_max) * 1.5  # Penalize too many
                else:
                    score = 0  # Perfect range

                if verbose:
                    if score == 0:
                        print(" ← Good range!")
                    elif num_detections < ideal_min:
                        print(" (too few)")
                    else:
                        print(" (too many)")

                # Keep track of best size
                if score < best_score:
                    best_score = score
                    best_size = test_size

            except Exception as e:
                if verbose:
                    print(f"  {test_size}px: Error - {e}")
                continue

        # Restore original values
        self.cell_size = original_cell_size
        self.step_size = original_step_size

        if verbose:
            print(f"\n✓ Recommended cell size: {best_size}px")

        return best_size

    def preview(self, img=np.array([])):
        if not len(img):
            img = self._img
        while True:
            cv2.imshow(self.WIN_NAME, img)
            key_pressed = cv2.waitKey(1)
            if key_pressed == ord('q'):
                # Quit
                break

if __name__ == '__main__':
    frame = BeeFrame()
    FILENAME = '003.png'
    PATH = '/home/chip/Dropbox/LITS/ML-003/dataset/processed_dataset/prespective_correction'

    cv2.namedWindow(frame.WIN_NAME, cv2.WINDOW_NORMAL)

    frame.load_image(PATH, FILENAME)
    frame.image.hitogram_normalization()
    frame.image.blur()
    #frame.get_cell_size()

    frame.image.add_weighted(mask=np.zeros( \
        (frame.image.height, frame.image.width, frame.image.NUM_CHANNELS), dtype=np.uint8))

    frame.preview()

    cv2.destroyAllWindows()