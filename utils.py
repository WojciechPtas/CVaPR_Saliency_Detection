import os
from random import sample
from typing import List
from pathlib import Path
import cv2
import sys

import numpy as np
import pandas as pd


def eprint(*args, **kwargs):
    print(*args, file = sys.stderr, **kwargs)


def load_image(path):
    img = cv2.imread(str(path))
    (h, w, c) = img.shape[:3]
    return (img, h, w, c)


def load_binary_image_and_resize(path, shape) -> np.array:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, shape)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = np.array(image > 1).astype(int)
    return image


def choose_random_images_from_path(path: str, number_of_images: int) -> List[str]:
    all_images = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    if len(all_images) < number_of_images:
        raise ValueError("Number of files is lower than number of files you want select")
    return sample(all_images, number_of_images)


def get_other_map(image_list: List):
    result = image_list[0]
    for image in image_list[1:]:
        result = cv2.bitwise_or(result, image)
    return result


def save_result_to_excel(result: dict, output_path: str, excel_file_name: str = "result.xlsx"):
    excel_path = os.path.join(output_path, excel_file_name)

    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([result])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        df_combined.to_excel(excel_path, index=False)
    else:
        df_new = pd.DataFrame([result])
        df_new.to_excel(excel_path, index=False)

def show_image(image):
    cv2.imshow("Result", image)
    cv2.waitKey(0)
