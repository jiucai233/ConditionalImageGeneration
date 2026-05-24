import cv2
import numpy as np

def color_transfer(src, ref, mask):
    """
    Transfers the color characteristics (LAB space) of ref to src,
    specifically focusing on the skin area immediately surrounding the eyebrow mask.
    """
    # Dilate the eyebrow mask further to define a local skin band surrounding the eyebrows
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    skin_mask = (dilated_mask > 0) & (mask == 0)
    
    if not np.any(skin_mask):
        skin_mask = (mask == 0) # Fallback to full background
        
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    for i in range(3):
        src_channel = src_lab[:, :, i]
        ref_channel = ref_lab[:, :, i]
        mean_src, std_src = src_channel[skin_mask].mean(), src_channel[skin_mask].std()
        mean_ref, std_ref = ref_channel[skin_mask].mean(), ref_channel[skin_mask].std()
        if std_src > 1e-5:
            src_lab[:, :, i] = (src_channel - mean_src) * (std_ref / std_src) + mean_ref
        else:
            src_lab[:, :, i] = src_channel - mean_src + mean_ref
            
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
