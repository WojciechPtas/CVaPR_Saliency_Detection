#!/usr/bin/python3
import cv2
import argparse
import sys
import numpy as np
from pathlib import Path

def eprint(*args, **kwargs):
    print(*args, file = sys.stderr, **kwargs)

def load_image(path):
    img = cv2.imread(str(path))
    (h, w, c) = img.shape[:3]
    return (img, h, w, c)

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


def main():
    parser = argparse.ArgumentParser(
        prog = "saliency",
        description = "Saliency Detection: A Boolean Map Approach",
    )
    
    parser.add_argument('input_path', type = Path)
    parser.add_argument('-o', '--output_path', type = Path)
    parser.add_argument('-d', '--debug_dir', type = Path)
    parser.add_argument('--opening_kernel', type = int, default = 13)
    parser.add_argument('--threshold_step', type = int, default = 8)
    args = parser.parse_args()
    
    (img, img_h, img_w, img_c) = load_image(args.input_path)
    eprint(f"image_width: {img_w}")
    eprint(f"image_height: {img_h}")
    eprint(f"opening_kernel_size: {args.opening_kernel}")
    eprint(f"threshold_step: {args.threshold_step}")
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Compute (and optionally write) boolean maps
    bool_maps = compute_boolean_maps(img_lab, args.threshold_step, args.opening_kernel)
    if args.debug_dir:
        for (i, bool_map_img) in enumerate(bool_maps):
            path = args.debug_dir / f"bool_{i}.png"
            eprint(f"Saving boolean map to {path}")
            cv2.imwrite(str(path), bool_map_img)
            
    
    # TODO further processing
    result_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    if args.output_path is None:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(str(args.output_path), result_img)

if __name__ == '__main__':
    main()

