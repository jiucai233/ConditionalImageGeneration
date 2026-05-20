import cv2
import numpy as np

def dilate_mask(mask, pixels=5):
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)
