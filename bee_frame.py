import os

import cv2
import numpy as np

class BeeFrame(object):
    """"""
    WIN_NAME = 'Bee Frame'

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

        def draw_circle(self, x, y, color):
            cv2.circle(img=self._img,
                       center=(    x + self.cell_size // 2, \
                                   y + self.cell_size // 2), 
                       radius=int(self.cell_size / 1.8),
                       color=color,
                       thickness=2)

        def draw_rect(self, x, y, color):
            cv2.rectangle(img=self._img, 
                          pt1=(x, y), 
                          pt2=(x + self.cell_size, y + self.cell_size), 
                          color=color, 
                          thickness=2)
            
        def draw_dot(self, x, y, color):
            cv2.putText(img=self._img, 
                        text='.', 
                        org=(x, y + self.cell_size),  # TOP LEFT corner if bottomLeftOrigin=False
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=4, 
                        color=color, 
                        thickness=1, 
                        bottomLeftOrigin=False)

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

    
    def __init__(self):
        """Tool to measure the averange cell size and starting x, y coordinats."""
        self.cell_size = 60
        self.step_size = 30
        self.start_x = 0
        self.start_y = 0


    def load_image(self, path, file_name):
        if hasattr(self, 'image'):
            del self.image
        self.image = self.Image(path, file_name)

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
                self.step_size = self.cell_size // 2
                self.start_x, self.start_y = data['ix_iy']
                break
        cv2.destroyAllWindows()


    
    def sliding_window(self, start_x=0, start_y=0, use_pattern=False):
        """Slide a window across the image."""
        odd_step = False
        for y in xrange(start_y, self.image.height, self.step_size):
            # Shift sliding window to the left on odd step to match chess board pattern.
            if use_pattern and odd_step:
                curr_x = start_x + self.step_size // 2
            else:
                curr_x = start_x
            odd_step = not odd_step

            for x in xrange(curr_x, self.image.width, self.step_size):
                # yield the current window
                window = self.image._img[y:y + self.cell_size, \
                                    x:x + self.cell_size]
                if window.shape[0] == window.shape[1]:
                    yield (x, y, window)

    def preview(self):
        while True:
            cv2.imshow(self.WIN_NAME, self.image._img)
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