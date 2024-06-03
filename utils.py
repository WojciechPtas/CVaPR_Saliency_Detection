import cv2
import sys


def eprint(*args, **kwargs):
    print(*args, file = sys.stderr, **kwargs)


def load_image(path):
    img = cv2.imread(str(path))
    (h, w, c) = img.shape[:3]
    return (img, h, w, c)
