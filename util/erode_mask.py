import cv2
import numpy as np

def erode_mask(mask, pixels=5):
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)
