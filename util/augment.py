import cv2
import numpy as np

def augment_image_and_mask(img, mask, horizontal_flip=True, color_jitter=True):
    """
    Applies data augmentation to both the image and the mask simultaneously.
    - img: numpy array of shape (H, W, 3) in RGB format.
    - mask: numpy array of shape (H, W) in grayscale format.
    """
    # 1. Random Horizontal Flip (50% probability)
    if horizontal_flip and np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        
    # 2. Slight Color Jitter (only on img, 50% probability)
    if color_jitter and np.random.rand() > 0.5:
        # Contrast adjustment range: [0.9, 1.1]
        alpha = np.random.uniform(0.9, 1.1)
        # Brightness adjustment range: [-15, 15]
        beta = np.random.randint(-15, 16)
        
        # Apply jitter and clip to stay in valid range [0, 255]
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        
    return img, mask

def get_random_zoom_crop_info(mask, image_shape, min_padding=1.8, max_padding=2.6, max_shift=15):
    """
    Calculates a randomized crop box (x1, y1, side, side) around the mask's bounding box.
    Provides random zoom (padding ratio variation) and spatial translation (shift) to boost spatial robustness.
    """
    h, w = image_shape[:2]
    y_indices, x_indices = np.where(mask > 0)
    
    if len(x_indices) == 0:
        # Fallback to standard center crop if no mask foreground exists
        side = min(h, w)
        return (w // 2 - side // 2, h // 2 - side // 2, side, side)
    
    # 1. Calculate bounding box of the eyebrow mask
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    bw = max_x - min_x
    bh = max_y - min_y
    
    # 2. Randomize zoom scale via padding ratio
    padding_ratio = np.random.uniform(min_padding, max_padding)
    side = int(max(bw, bh) * padding_ratio)
    
    # 3. Randomize center position with spatial shift (translation)
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    cx += np.random.randint(-max_shift, max_shift + 1)
    cy += np.random.randint(-max_shift, max_shift + 1)
    
    # 4. Determine top-left corner
    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)
    
    # 5. Handle boundaries and adjust to ensure it remains inside the image
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    if x1 + side > w:
        x1 = max(0, int(w - side))
        side = min(side, w)
    if y1 + side > h:
        y1 = max(0, int(h - side))
        side = min(side, h)
        
    return (x1, y1, int(side), int(side))
