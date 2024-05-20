#!/usr/bin/python3
import cv2
import argparse
import sys
from pathlib import Path

def eprint(*args, **kwargs):
    print(*args, file = sys.stderr, **kwargs)

def load_image(path):
    img = cv2.imread(str(path))
    (h, w, c) = img.shape[:3]
    return (img, h, w, c)

def compute_boolean_maps(image, threshold_step = 8):
    return [] # TODO


def main():
    parser = argparse.ArgumentParser(
        prog = "saliency",
        description = "Saliency Detection: A Boolean Map Approach",
    )
    
    parser.add_argument('input_path', type = Path)
    parser.add_argument('-o', '--output_path', type = Path)
    parser.add_argument('-d', '--debug_dir', type = Path)
    args = parser.parse_args()
    
    (img, img_h, img_w, img_c) = load_image(args.input_path)
    eprint(f"image_width: {img_w}")
    eprint(f"image_height: {img_h}")
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # TODO processing
    result_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    if args.output_path is None:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(str(args.output_path), result_img)

if __name__ == '__main__':
    main()

