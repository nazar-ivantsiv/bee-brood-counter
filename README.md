Bee Brood Counter
=================

Pipeline for detecting and counting bee brood caps on a pre-processed photo of bee frame.

Pre-processing includes cropping with perspective correction on original photo.

#### To run:
- install required packages (requirements.txt)
    - Python 2.7
    - TensorFlow (>=r0.7)
    - NumPy
    - Jupyter Notebook
- extract dataset.zip content into ./dataset/ folder and train ConvNet.
- change PATH and FILENAME constants to point to 'bee_frame_sample.png' file location 
- use ConvNet to detect and count capped bee brood cells.

![alt text](https://github.com/nazar-ivantsiv/bee-brood-counter/blob/master/detection_result.png "Intermediate result of detection")
