import cv2
import numpy as np

def resize_and_pad(image, target_size=512):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    if len(image.shape) == 2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_img = np.zeros((target_size, target_size), dtype=image.dtype)
    else:
        resized = cv2.resize(image, (new_w, new_h))
        pad_img = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
        
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    pad_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return pad_img, (x_off, y_off, new_w, new_h), scale
