Bee Brood Counter
=================

Pipeline for detecting and counting bee brood caps on a pre-processed photo of bee frame.

Pre-processing includes cropping with perspective correction on original photo.

#### Requirements:
- Python 3.11+
- TensorFlow 2.15+
- OpenCV 4.8+
- NumPy, SciPy, scikit-learn
- Jupyter Notebook

#### To run:
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Extract dataset.zip content into `./dataset/` folder

3. Open and run `bee_brood_counter.ipynb` in Jupyter Notebook to train the ConvNet

4. Change PATH and FILENAME constants to point to your bee frame image location

5. Use the trained ConvNet to detect and count capped bee brood cells

#### Migration Notes:
This application has been modernized from Python 2.7 to Python 3.11+ with the following major changes:
- Upgraded TensorFlow from r0.7 to 2.15+ (using Keras API)
- Upgraded OpenCV from 3.0.0 to 4.8+
- Updated all Python 2 syntax to Python 3 (removed tuple unpacking in parameters, replaced xrange with range)
- Rewrote model using TensorFlow 2.x Keras functional API
- Model files now use `.keras` extension instead of `.ckpt`

![alt text](https://github.com/nazar-ivantsiv/bee-brood-counter/blob/master/detection_result.png "Intermediate result of detection")
