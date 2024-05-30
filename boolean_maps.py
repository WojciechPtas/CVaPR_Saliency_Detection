import cv2
import numpy as np

def compute_channel_boolean_maps(img, threshold_step, opening_kernel_size):
    # We treat them separately so that appear more sorted
    maps = []
    inv_maps = []
    
    for threshold in range(0, 255, threshold_step):
        bm = (img < threshold) * np.uint8(255)
        inv = (img >= threshold) * np.uint8(255)
        
        if opening_kernel_size != 0:
            kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
            bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, kernel)
            inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
        
        maps.append(bm)
        inv_maps.append(inv)

    return maps + inv_maps

def compute_boolean_maps(img, threshold_step, opening_kernel_size):
    (img_h, img_w, img_c) = img.shape[:3]
    (l, a, b) = cv2.split(img)
    bool_maps = []
    for channel in [l, a, b]:
        bool_maps += compute_channel_boolean_maps(channel, threshold_step, opening_kernel_size)
    return bool_maps