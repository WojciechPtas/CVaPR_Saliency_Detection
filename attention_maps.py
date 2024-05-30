import cv2
import numpy as np


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

def calculate_attention_map(boolean_maps):
    attention_map = np.zeros(boolean_maps[0].shape, dtype=np.float64)
    i = 0
    for boolean_map in boolean_maps:
        activation = activate_map(boolean_map)
        path = f"results/attn_{i}.png"
        cv2.imwrite(str(path), activation)
        i += 1
        attention_map += activation
    attention_map = attention_map / len(boolean_maps)
    return cv2.convertScaleAbs(attention_map)