import cv2

def restore_from_pad(pad_img, crop_info, orig_shape):
    x_off, y_off, new_w, new_h = crop_info
    cropped = pad_img[y_off:y_off+new_h, x_off:x_off+new_w]
    return cv2.resize(cropped, (orig_shape[1], orig_shape[0]))
