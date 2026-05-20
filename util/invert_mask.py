import cv2
import numpy as np

def invert_mask(mask):
    """
    Inverts a binary or grayscale mask.
    Assumes white (255) becomes black (0) and vice versa.
    """
    return 255 - mask
