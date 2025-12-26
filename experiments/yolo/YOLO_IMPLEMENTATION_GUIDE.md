# YOLOv8 Implementation Guide for Bee Brood Counting

## Overview

This guide will help you transition from the CNN sliding window approach to YOLOv8 object detection for more accurate and faster brood cell counting.

---

## Why YOLOv8 is Better

### Current CNN Approach Problems:
- ❌ Overfits to bright spots (color/texture bias)
- ❌ Sliding window is slow (thousands of windows per image)
- ❌ Requires manual cell size calibration
- ❌ Many overlapping detections even with NMS
- ❌ Can't distinguish cell vs cell fragment

### YOLOv8 Advantages:
- ✅ Learns complete cell structure (shape + context)
- ✅ 100x faster inference (single pass through image)
- ✅ Handles multiple scales automatically
- ✅ Better with overlapping/touching cells
- ✅ State-of-the-art accuracy

---

## Prerequisites

### Software Installed:
- ✅ Python 3.9
- ✅ ultralytics (YOLOv8) - installed
- ✅ OpenCV - installed

### What You Need:
1. **Full bee frame images** (not 60x60 patches)
2. **Bounding box annotations** around each brood cell
3. **~50-200 annotated images** for good results

---

## Step 1: Collect Full Frame Images

You need full bee frame images (like `bee_frame_sample.png`).

**Recommended**:
- 50-100 diverse frames minimum
- 200+ frames for production quality
- Include variety: different lighting, angles, bee coverage

**Sources**:
- Your own bee frame photos
- Public datasets (BeeImage, PollenDataset)
- Video frame extraction from hive cameras

---

## Step 2: Annotate Images with Bounding Boxes

You need to draw rectangles around each brood cell in your images.

### Option A: LabelImg (Recommended - Free, Easy)

```bash
# Install LabelImg
pip install labelImg

# Run annotation tool
labelImg
```

**Instructions**:
1. Click "Open Dir" → Select folder with your bee frames
2. Click "Change Save Dir" → Select output folder
3. Set format to "YOLO" (not Pascal VOC)
4. For each image:
   - Press 'W' to create box
   - Draw rectangle around ONE brood cell
   - Type label: "brood"
   - Repeat for all visible brood cells
   - Press 'D' for next image

**Output**: Creates `.txt` annotation files in YOLO format

### Option B: Roboflow (Web-based, Free Tier)

1. Go to https://roboflow.com
2. Create project → "Object Detection"
3. Upload images
4. Use box tool to annotate cells
5. Export in "YOLOv8" format

###Option C: CVAT (Advanced, Self-hosted)

For larger datasets, use CVAT (Computer Vision Annotation Tool)

---

## Step 3: Dataset Format

YOLO requires this structure:

```
bee_brood_yolo/
├── data.yaml              # Dataset config
├── train/
│   ├── images/
│   │   ├── frame001.jpg
│   │   ├── frame002.jpg
│   │   └── ...
│   └── labels/
│       ├── frame001.txt   # Annotations for frame001.jpg
│       ├── frame002.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/  (optional)
    ├── images/
    └── labels/
```

### Annotation Format (.txt files):

Each line in a label file:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to 0-1:
```
0 0.5 0.3 0.028 0.055
0 0.52 0.31 0.029 0.054
```

Where:
- `class_id`: 0 (for "brood" class)
- `x_center, y_center`: Center of bounding box (0-1, relative to image size)
- `width, height`: Box dimensions (0-1, relative to image size)

---

## Step 4: Create data.yaml

```yaml
# Path to dataset (can be relative)
path: ./bee_brood_yolo

# Train/val/test splits
train: train/images
val: val/images
test: test/images  # optional

# Class names
names:
  0: brood
```

---

## Step 5: Train YOLOv8

```python
from ultralytics import YOLO

# Load pretrained model (nano version for speed)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt for better accuracy

# Train on your dataset
results = model.train(
    data='bee_brood_yolo/data.yaml',
    epochs=50,              # Start with 50, increase if needed
    imgsz=640,              # Image size
    batch=16,               # Reduce if out of memory (try 8, 4)
    device='cpu',           # Use 'mps' for M1 GPU (if not crashing)
    workers=4,
    patience=10,            # Early stopping
    project='runs/brood',
    name='yolov8_brood_v1'
)

# Training takes ~1-3 hours on CPU for 50 epochs
```

---

## Step 6: Validate Model

```python
# Validate on test set
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")      # Mean Average Precision @ IoU 0.5
print(f"mAP50-95: {metrics.box.map}")     # mAP at IoU 0.5:0.95
print(f"Precision: {metrics.box.mp}")     # Precision
print(f"Recall: {metrics.box.mr}")        # Recall
```

**Target Metrics**:
- mAP50 > 0.80 (good)
- mAP50 > 0.90 (excellent)
- Precision > 0.85
- Recall > 0.85

---

## Step 7: Run Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/brood/yolov8_brood_v1/weights/best.pt')

# Run detection on new image
results = model.predict(
    'bee_frame_sample.png',
    conf=0.25,          # Confidence threshold
    iou=0.45,           # NMS IoU threshold
    device='cpu'
)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Box coordinates
        conf = box.conf[0]             # Confidence
        cls = box.cls[0]               # Class (0 = brood)

        print(f"Brood cell at ({x1}, {y1}, {x2}, {y2}) - confidence: {conf:.2f}")

# Save visualization
annotated = results[0].plot()  # Draw boxes on image
cv2.imwrite('yolo_detection_result.png', annotated)

# Get count
total_cells = len(results[0].boxes)
print(f"Total brood cells detected: {total_cells}")
```

---

## Step 8: Integration with BeeFrame

I'll create a new `BeeFrameYOLO` class that wraps YOLOv8:

```python
class BeeFrameYOLO:
    def __init__(self, model_path='best.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect_brood_cells(self, image_path, conf=0.25):
        results = self.model.predict(image_path, conf=conf, device='cpu')
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])

            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence
            })

        return detections

    def get_detection_summary(self, detections):
        return {
            'total_detections': len(detections),
            'avg_confidence': np.mean([d['confidence'] for d in detections]),
            'max_confidence': np.max([d['confidence'] for d in detections]),
            'min_confidence': np.min([d['confidence'] for d in detections])
        }
```

---

## Quick Start: Bootstrap Training Data

If you don't have annotated data yet, here's a bootstrap approach:

### Use your CNN to create initial annotations:

```python
# Run CNN detection on full frame
python run_brood_counter.py --image frame.jpg --cell-size 30 --cpu-only

# This creates bounding boxes
# Export to YOLO format
# Manually review and fix errors in LabelImg
# Train YOLO with these corrected annotations
```

---

## Annotation Tips

1. **Be consistent**: Use similar box sizes for similar cells
2. **Annotate visible cells only**: Skip partially visible or uncertain ones
3. **Include negatives**: Annotate some frames with NO brood cells (empty frames)
4. **Quality > Quantity**: 50 well-annotated images > 200 sloppy ones
5. **Start small**: Annotate 20 images, train, test, then annotate more

---

## Expected Results

### CNN Sliding Window (current):
- Detection time: 30-60 seconds per frame
- Accuracy: ~85% (overfits to brightness)
- Detections: Varies wildly with cell_size parameter
- False positives: High on bright spots

### YOLOv8 (after training):
- Detection time: 0.5-2 seconds per frame
- Accuracy: 92-98% (learns structure)
- Detections: Consistent across images
- False positives: Low (learns context)

---

## Next Steps

1. Gather 50-100 full bee frame images
2. Annotate using LabelImg (label: "brood")
3. Split into train (80%) / val (20%)
4. Run training script (see Step 5)
5. Evaluate results
6. If mAP < 0.80, annotate 50 more images and retrain

---

## Troubleshooting

### "Out of memory during training"
```python
# Reduce batch size
results = model.train(data='data.yaml', batch=4)
```

### "Model not learning (mAP stays low)"
- Check annotations are correct (view in LabelImg)
- Need more diverse training images
- Increase epochs to 100
- Try larger model: `yolov8s.pt` or `yolov8m.pt`

### "Too many false positives"
```python
# Increase confidence threshold
results = model.predict('frame.jpg', conf=0.5)  # instead of 0.25
```

### "Missing many cells (low recall)"
```python
# Lower confidence threshold
results = model.predict('frame.jpg', conf=0.15)
# Or train longer / add more training data
```

---

## Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **LabelImg Download**: https://github.com/HumanSignal/labelImg
- **Roboflow Tutorial**: https://roboflow.com/annotate
- **YOLO Training Tutorial**: https://docs.ultralytics.com/modes/train/

---

## Files to Create

I will create the following helper scripts for you:

1. `prepare_yolo_dataset.py` - Convert CNN detections to YOLO annotations (bootstrap)
2. `train_yolo.py` - Training script with best practices
3. `run_yolo_detection.py` - Inference script for production use
4. `bee_frame_yolo.py` - YOLOv8-based BeeFrame class

---

**Status**: YOLOv8 installed ✅
**Next**: Annotate images or run bootstrap script to generate initial training data
