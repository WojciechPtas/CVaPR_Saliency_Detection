#!/usr/bin/python3
import cv2
import argparse
from pathlib import Path
from utils import eprint, load_image
from saliency import calculate_saliency_map

def main():
    parser = argparse.ArgumentParser(
        prog = "saliency",
        description = "Saliency Detection: A Boolean Map Approach",
    )
    
    parser.add_argument('input_path', type = Path)
    parser.add_argument('-m', '--input_mask', type=Path)
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
    
    result_img = calculate_saliency_map(img_lab, args.threshold_step, args.opening_kernel, args.debug_dir)


    if args.output_path is None:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(str(args.output_path), result_img)

if __name__ == '__main__':
    main()
