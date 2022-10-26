from PIL import Image

import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import torch
from torch import device, nn

device = "mps"

### Own imports
from DETR import main, plot_box  # DETR.py
from Blur import background_blurrer  # Blur.py


def blur_dataset(path: str) -> None:
    """Blurs images of an entire directory

    Args:
        path (str): path to the directory
    """
    img_formats = ["jpg", "JPG", "jpeg", "JPEG"]
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlypic = [f for f in onlyfiles if f.split(".")[-1] in img_formats]
    # create blur subdirectory

    if not os.path.exists(join(path, "blur/")):
        os.mkdir(join(path, "blur/"))
    imgpath = join(path, "blur/")
    for img in tqdm(onlypic):
        ### use object detection
        scores, boxes = main(join(path, img))

        # open PIL.Image
        pil_img = Image.open(join(path, img))
        # save OD image with _od end tag next to normal images
        plot_box(
            pil_img,
            scores,
            boxes,
            join(imgpath, img.split(".")[0] + "_od." + img.split(".")[1]),
        )

        ## apply background blur
        blurred_img_dct = background_blurrer(pil_img, boxes)

        img_blurred = Image.fromarray(blurred_img_dct["masked"], "RGB")
        img_blurred.save(join(imgpath, img))


if __name__ == "__main__":
    p = "imagenette2-320/train/n02102040"
    blur_dataset(p)
