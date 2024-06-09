#!/usr/bin/python3
import os

import cv2
import argparse
from pathlib import Path
from threading import Thread

from shuffled_auc import auc_shuffled
from utils import eprint, load_image, load_binary_image_and_resize, choose_random_images_from_path, get_other_map, \
    save_result_to_excel, show_image
from saliency import calculate_saliency_map

# Example use
# python main.py .\img\1.jpg -m .\img\Test4\1_result.png --random_images_dir .\img\Test4\ -o .\img\archive\


def main():
    parser = argparse.ArgumentParser(
        prog = "saliency",
        description = "Saliency Detection: A Boolean Map Approach",
    )
    
    parser.add_argument('input_path', type = Path)
    parser.add_argument('-m', '--input_mask', type=Path)
    parser.add_argument('-o', '--output_path', type = Path)
    parser.add_argument('-d', '--debug_dir', type = Path)
    parser.add_argument('--random_images_dir', type=Path)
    parser.add_argument('--opening_kernel', type = int, default = 13)
    parser.add_argument('--threshold_step', type = int, default = 8)
    parser.add_argument('--auc_n_splits', type=int, default=100)
    parser.add_argument('--auc_step_size', type=float, default=0.1)
    parser.add_argument('--excel_file_name', type=Path, default='result.xlsx')
    args = parser.parse_args()
    
    (img, img_h, img_w, img_c) = load_image(args.input_path)
    eprint(f"image_width: {img_w}")
    eprint(f"image_height: {img_h}")
    eprint(f"opening_kernel_size: {args.opening_kernel}")
    eprint(f"threshold_step: {args.threshold_step}")
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    result_img = calculate_saliency_map(img_lab, args.threshold_step, args.opening_kernel, args.debug_dir)

    if args.output_path is None:
        show_image_thread = Thread(target=show_image, args=(result_img,))
        show_image_thread.setDaemon(True)
        show_image_thread.start()
    else:
        save_path = os.path.join(str(args.output_path), "result_" + str(args.input_path.name))
        cv2.imwrite(save_path, result_img)

    if args.input_mask is None:
        print("Missing fixation map path")
        exit(1)

    if args.random_images_dir is None:
        print("Missing path to random images")
        exit(1)

    shape = (img_w, img_h)
    fixation_map = load_binary_image_and_resize(args.input_mask, shape)

    random_images_paths = choose_random_images_from_path(args.random_images_dir, 3)
    random_images = [load_binary_image_and_resize(path, shape) for path in random_images_paths]
    other_map = get_other_map(random_images)

    average_auc = auc_shuffled(result_img, fixation_map, other_map,
                               n_splits=args.auc_n_splits,
                               step_size=args.auc_step_size, to_plot=True)

    result_data = {
        "file": args.input_path,
        "average_auc": average_auc,
        "opening_kernel": args.opening_kernel,
        "threshold_step": args.threshold_step,
        "auc_n_splits": args.auc_n_splits,
        "auc_step_size": args.auc_step_size,
        "image_width": img_w,
        "image_height": img_h
    }

    if args.output_path is None:
        output_path = ""
    else:
        output_path = args.output_path

    save_result_to_excel(result_data, output_path, args.excel_file_name)


if __name__ == '__main__':
    main()
