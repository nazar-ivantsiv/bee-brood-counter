# Experiments Archive

This folder contains experimental approaches that were explored for bee brood counting but are not the primary production solution.

## Folders

### `cnn_sliding_window/`
**Approach**: Sliding window CNN classifier
**Status**: Archived - replaced by density estimation
**Issues**:
- Slow inference (30-180 seconds per frame)
- Overfits to brightness
- Requires manual cell size calibration
- Many false positives

**Contents**:
- `run_brood_counter.py` - Main detection script
- `bee_frame.py` - BeeFrame class with CNN integration
- `best_baseline_model.keras` - Trained CNN model (72K parameters, 95% accuracy)
- `demo_detection.py` - Demo detection script
- `hyperparameter_tuning/` - Keras Tuner results
- `evaluation_results/` - Model evaluation metrics
- Documentation files

**Results**:
- Accuracy: ~85-90% (overfits to bright spots)
- Speed: 30-180 seconds per frame
- Count on bee_frame_sample.png: 54-259 (varies with cell_size parameter)

### `yolo/`
**Approach**: YOLOv8 object detection
**Status**: Planned but not implemented
**Why not pursued**: Requires annotating bounding boxes (slow), density maps are better for counting

**Contents**:
- `YOLO_IMPLEMENTATION_GUIDE.md` - Implementation guide

---

## Current Production Approach

See main project folder for **Density Estimation** approach:
- Fast inference (0.5-2 seconds)
- No brightness bias
- Easy annotation (point clicks)
- Accurate counting (95-98%)
- Beautiful heatmap visualizations
