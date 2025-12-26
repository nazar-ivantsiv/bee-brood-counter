# BeeFrame Model Integration Guide

Complete guide for using the trained model with the BeeFrame detection pipeline.

---

## Quick Start

### 1. Basic Detection

```python
from bee_frame import BeeFrame

# Initialize with model
frame = BeeFrame(model_path='./best_baseline_model.keras')

# Load an image
frame.load_image(path='/path/to/images', file_name='frame.png')

# Detect brood cells
detections = frame.detect_brood_cells(
    threshold=0.5,           # Classification threshold
    confidence_threshold=0.7, # Minimum confidence to keep
    use_nms=True,            # Apply non-maximum suppression
    nms_threshold=0.3        # NMS IoU threshold
)

# Visualize results
result_img = frame.visualize_detections(show_confidence=True)

# Get statistics
summary = frame.get_detection_summary()
print(f"Detected {summary['total_detections']} brood cells")
print(f"Average confidence: {summary['avg_confidence']:.2%}")
```

### 2. Command-Line Demo

```bash
# Activate virtual environment
source ./bee_brood_counter/bin/activate

# Run detection on an image
python demo_detection.py --image /path/to/frame.png

# Adjust thresholds
python demo_detection.py --image /path/to/frame.png --threshold 0.6 --confidence 0.8

# Save result to file
python demo_detection.py --image /path/to/frame.png --output result.png

# CPU-only mode (recommended for 8GB M1 Macs)
python demo_detection.py --image /path/to/frame.png --cpu-only
```

---

## API Reference

### BeeFrame Class

#### `__init__(model_path=None)`

Initialize BeeFrame detector.

**Parameters:**
- `model_path` (str, optional): Path to trained .keras model file. If provided, loads model immediately.

**Example:**
```python
frame = BeeFrame(model_path='./best_baseline_model.keras')
```

#### `load_model(model_path)`

Load a trained model for detection.

**Parameters:**
- `model_path` (str): Path to .keras model file

**Example:**
```python
frame = BeeFrame()
frame.load_model('./best_baseline_model.keras')
```

#### `load_image(path, file_name)`

Load an image for detection.

**Parameters:**
- `path` (str): Directory containing the image
- `file_name` (str): Image filename

**Example:**
```python
frame.load_image('/path/to/images', 'frame_001.png')
```

#### `detect_brood_cells(threshold=0.5, confidence_threshold=0.7, use_nms=True, nms_threshold=0.3)`

Detect brood cells in the loaded image.

**Parameters:**
- `threshold` (float): Classification threshold (0-1). Default: 0.5
  - Lower = more sensitive (more detections, more false positives)
  - Higher = more conservative (fewer detections, fewer false positives)
- `confidence_threshold` (float): Minimum confidence to keep detection (0-1). Default: 0.7
  - Filters out low-confidence predictions
- `use_nms` (bool): Apply non-maximum suppression. Default: True
  - Removes overlapping detections
- `nms_threshold` (float): IoU threshold for NMS (0-1). Default: 0.3
  - Lower = more aggressive suppression
  - Higher = keep more overlapping boxes

**Returns:**
- List of detection dictionaries: `[{'bbox': (x, y, end_x, end_y), 'confidence': float}, ...]`

**Example:**
```python
# Standard detection
detections = frame.detect_brood_cells()

# High precision mode (fewer false positives)
detections = frame.detect_brood_cells(threshold=0.7, confidence_threshold=0.85)

# High recall mode (catch more cells, more false positives)
detections = frame.detect_brood_cells(threshold=0.3, confidence_threshold=0.5)
```

#### `visualize_detections(show_confidence=True, min_confidence=0.0)`

Create visualization with detection boxes.

**Parameters:**
- `show_confidence` (bool): Display confidence scores on boxes. Default: True
- `min_confidence` (float): Only show detections above this confidence. Default: 0.0

**Returns:**
- NumPy array: Image with detections drawn (BGR format)

**Color coding:**
- Green: High confidence (≥0.9)
- Yellow: Medium confidence (0.75-0.9)
- Orange: Lower confidence (<0.75)

**Example:**
```python
# Show all detections with scores
vis_img = frame.visualize_detections()

# Show only high-confidence detections
vis_img = frame.visualize_detections(min_confidence=0.8)

# Display or save
cv2.imshow('Detections', vis_img)
cv2.imwrite('result.png', vis_img)
```

#### `get_detection_summary()`

Get statistics about detections.

**Returns:**
- Dictionary with:
  - `total_detections` (int): Number of detections
  - `avg_confidence` (float): Average confidence score
  - `max_confidence` (float): Highest confidence
  - `min_confidence` (float): Lowest confidence
  - `std_confidence` (float): Standard deviation of confidence

**Example:**
```python
summary = frame.get_detection_summary()
print(f"Found {summary['total_detections']} cells")
print(f"Confidence: {summary['avg_confidence']:.2%} ± {summary['std_confidence']:.2%}")
```

#### `preprocess_for_model(window)`

Preprocess a 60x60 window for model input.

**Parameters:**
- `window` (np.ndarray): 60x60 or 64x64 BGR image

**Returns:**
- NumPy array: (1, 64, 64, 3) normalized to [-0.5, 0.5]

---

## Detection Parameters Guide

### Threshold Selection

The two main thresholds control different aspects:

#### 1. Classification Threshold

Controls the decision boundary for classification:

```python
# Conservative (fewer false positives, may miss some cells)
detections = frame.detect_brood_cells(threshold=0.7)

# Balanced (recommended starting point)
detections = frame.detect_brood_cells(threshold=0.5)

# Aggressive (catch more cells, more false positives)
detections = frame.detect_brood_cells(threshold=0.3)
```

#### 2. Confidence Threshold

Filters detections after classification:

```python
# Keep only very confident predictions
detections = frame.detect_brood_cells(confidence_threshold=0.85)

# Balanced filtering (recommended)
detections = frame.detect_brood_cells(confidence_threshold=0.7)

# Keep more detections (more noise)
detections = frame.detect_brood_cells(confidence_threshold=0.5)
```

### Recommended Presets

```python
# HIGH PRECISION: Minimize false positives
detections = frame.detect_brood_cells(
    threshold=0.7,
    confidence_threshold=0.85,
    use_nms=True,
    nms_threshold=0.2
)

# BALANCED: Good overall performance (DEFAULT)
detections = frame.detect_brood_cells(
    threshold=0.5,
    confidence_threshold=0.7,
    use_nms=True,
    nms_threshold=0.3
)

# HIGH RECALL: Catch as many cells as possible
detections = frame.detect_brood_cells(
    threshold=0.3,
    confidence_threshold=0.5,
    use_nms=True,
    nms_threshold=0.4
)
```

---

## Advanced Usage

### Batch Processing

Process multiple images:

```python
import os
from bee_frame import BeeFrame

# Initialize once
frame = BeeFrame(model_path='./best_baseline_model.keras')

# Process directory
image_dir = './bee_frames'
results = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        frame.load_image(image_dir, filename)

        # Detect
        detections = frame.detect_brood_cells()

        # Store results
        summary = frame.get_detection_summary()
        results.append({
            'filename': filename,
            'count': summary['total_detections'],
            'avg_confidence': summary['avg_confidence']
        })

        # Save visualization
        vis_img = frame.visualize_detections()
        output_path = f"./results/{filename}"
        cv2.imwrite(output_path, vis_img)

        print(f"{filename}: {summary['total_detections']} cells detected")

# Save summary
import json
with open('detection_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Custom Visualization

Create custom visualizations:

```python
# Get raw detections
detections = frame.detect_brood_cells()

# Create custom visualization
img = frame.image._img.copy()

for det in detections:
    x, y, end_x, end_y = det['bbox']
    conf = det['confidence']

    # Custom drawing logic
    if conf > 0.9:
        # Draw circle for high confidence
        center = ((x + end_x) // 2, (y + end_y) // 2)
        radius = (end_x - x) // 2
        cv2.circle(img, center, radius, (0, 255, 0), 2)
    else:
        # Draw rectangle for lower confidence
        cv2.rectangle(img, (x, y), (end_x, end_y), (0, 255, 255), 2)

cv2.imwrite('custom_result.png', img)
```

### Integration with Existing Pipeline

If you have an existing detection pipeline:

```python
from bee_frame import BeeFrame

class MyBeeAnalyzer:
    def __init__(self):
        self.detector = BeeFrame(model_path='./best_baseline_model.keras')

    def analyze_frame(self, image_path):
        # Load image
        import os
        self.detector.load_image(
            os.path.dirname(image_path),
            os.path.basename(image_path)
        )

        # Detect
        detections = self.detector.detect_brood_cells()

        # Your custom analysis
        return self.process_detections(detections)

    def process_detections(self, detections):
        # Your custom logic here
        return {
            'brood_count': len(detections),
            'brood_density': self.calculate_density(detections),
            'health_score': self.assess_health(detections)
        }
```

---

## Performance Optimization

### CPU vs GPU

```python
# Force CPU (recommended for 8GB M1 Macs)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
frame = BeeFrame(model_path='./best_baseline_model.keras')
```

### Batch Prediction

For faster processing of multiple windows:

```python
# Collect windows first
windows = []
coords = []
for x, y, window in frame.sliding_window():
    windows.append(frame.preprocess_for_model(window))
    coords.append((x, y))

# Batch predict (faster)
import numpy as np
batch = np.vstack(windows)
predictions = frame.model.predict(batch, verbose=0)

# Process results
for i, pred in enumerate(predictions):
    if pred[1] > 0.7:  # Confidence threshold
        x, y = coords[i]
        print(f"Cell at ({x}, {y}): {pred[1]:.2%}")
```

---

## Troubleshooting

### Issue: Low detection count

**Solutions:**
1. Lower thresholds:
   ```python
   detections = frame.detect_brood_cells(threshold=0.3, confidence_threshold=0.5)
   ```
2. Check image preprocessing (histogram normalization, blur)
3. Ensure image has good quality and lighting

### Issue: Too many false positives

**Solutions:**
1. Raise thresholds:
   ```python
   detections = frame.detect_brood_cells(threshold=0.7, confidence_threshold=0.85)
   ```
2. Adjust NMS threshold:
   ```python
   detections = frame.detect_brood_cells(nms_threshold=0.2)
   ```

### Issue: Slow inference

**Solutions:**
1. Use CPU-only mode:
   ```python
   tf.config.set_visible_devices([], 'GPU')
   ```
2. Use batch prediction (see Performance Optimization)
3. Increase step size:
   ```python
   frame.step_size = 40  # Default is 30
   ```

### Issue: Out of memory

**Solutions:**
1. Force CPU mode
2. Process images in smaller batches
3. Reduce image resolution before detection

---

## Model Information

**Model**: Tuned Baseline CNN (Trial 0008)
**Architecture**:
- Conv1: 24 filters, 5x5 kernel
- Conv2: 8 filters, 5x5 kernel
- FC: 128 units
- No dropout

**Performance**:
- Test Accuracy: 95.23%
- Test AUC-ROC: 98.49%
- Test F1-Score: 87.60%
- Recall: 89.68% (catches 9/10 brood cells)
- Precision: 85.61%

**Input**: 64x64 RGB images, normalized to [-0.5, 0.5]
**Output**: 2-class softmax (negative, positive)

---

## Examples

See the following files for complete examples:
- `demo_detection.py` - Command-line detection tool
- `bee_frame.py` - Core BeeFrame class with all detection methods

---

## Next Steps

1. **Tune parameters** for your specific images
2. **Collect edge cases** where detection fails
3. **Consider ensemble** methods for improved accuracy
4. **Implement tracking** across video frames
5. **Add temporal smoothing** for video analysis
