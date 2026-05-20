import cv2

def smooth_mask(mask, ksize=15):
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)
