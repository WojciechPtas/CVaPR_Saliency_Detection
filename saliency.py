import cv2
import numpy as np
from utils import eprint
from boolean_maps import compute_boolean_maps


def save_maps(maps, path, name):
    for i, map in enumerate(maps):
        file_path = path / f"{name}_{i}.png"
        eprint(f"Saving {name} map to {file_path}")
        cv2.imwrite(str(file_path), map)


def activate_map(boolean_map):
    activation = np.array(boolean_map, dtype=np.uint8)
    mask_shape = (boolean_map.shape[0] + 2, boolean_map.shape[1] + 2)
    ffill_mask = np.zeros(mask_shape, dtype=np.uint8)
    
    for i in range(0, activation.shape[0]):
        for j in [0, activation.shape[1] - 1]:
            if activation[i,j]:
                cv2.floodFill(activation, ffill_mask, (j, i), 0)

    for i in [0, activation.shape[0] - 1]:
        for j in range(0, activation.shape[1]):
            if activation[i,j]:
                cv2.floodFill(activation, ffill_mask, (j, i), 0)
    return activation


def calculate_attention_map(boolean_maps, debug_dir=None):
    attention_map = np.zeros(boolean_maps[0].shape, dtype=np.float64)
    activated_maps = []
    for boolean_map in boolean_maps:
        activation = activate_map(boolean_map)
        activated_maps.append(activation)
        attention_map += activation
    if debug_dir:
        save_maps(activated_maps, debug_dir, "activated")
    attention_map = attention_map / len(boolean_maps)
    return cv2.convertScaleAbs(attention_map)


def post_proces_attention_map(attention_map):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    attention_map = cv2.dilate(attention_map, kernel, 1)    
    attention_map = cv2.normalize(attention_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return attention_map


def calculate_saliency_map(img, threshold_step, opening_kernel, debug_dir=None):
    boolean_maps = compute_boolean_maps(img, threshold_step, opening_kernel)
    if debug_dir:
        save_maps(boolean_maps, debug_dir, "bool")
    attention_map = calculate_attention_map(boolean_maps, debug_dir)
    post_processed = post_proces_attention_map(attention_map)
    return post_processed
