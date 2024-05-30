#!/usr/bin/python3
import cv2
import argparse
from pathlib import Path
from utils import eprint, load_image
from boolean_maps import compute_boolean_maps
from attention_maps import calculate_saliency_map



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
            
    
    # Calculate attention map
    attention_map = calculate_saliency_map(bool_maps)
    cv2.imshow("Attention Map", attention_map)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    attention_map = cv2.morphologyEx(attention_map, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening Map", attention_map)
    attention_map = cv2.morphologyEx(attention_map, cv2.MORPH_CLOSE,kernel)
    cv2.imshow("Close Map", attention_map)

    result_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    if args.output_path is None:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(str(args.output_path), result_img)

if __name__ == '__main__':
    main()

