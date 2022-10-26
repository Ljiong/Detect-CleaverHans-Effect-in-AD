from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'

import cv2
import numpy as np

import torch
from torch import device, nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy

from DETR import main, plot_box

torch.set_grad_enabled(False)
device = "cpu"

import pickle


def boxinator(boxes, image):
    h, w, c = np.array(image).shape
    boxes = np.around(boxes).astype(int)

    box = np.array([np.array(image)[:, :, 0] * 0] * len(boxes), dtype=bool)  # 3D tensor

    for i in range(len(boxes)):
        # now we are looking at box i
        xmin, ymin, xmax, ymax = np.array(np.round_(boxes[i]), dtype=int)

        box[i][ymin:ymax, xmin:xmax] = 1

    bbl_mask = np.array(image)[:, :, 0] * 0

    for mask in box:
        bbl_mask += mask
        bbl_mask[bbl_mask > 1] = 1
    bbl_mask[bbl_mask > 1] = 1
    return np.squeeze(bbl_mask)

    # bbl = bounding box list.


def background_blurrer(image, bbl):
    # make the bounding box list into a list of bounding box masks
    bbl_mask = boxinator(bbl.tolist(), image)

    # now we have a single mask masking the area that we don't want to blur

    temp_img = image
    temp_img = cv2.blur(
        np.array(temp_img).astype(np.uint8), (20, 20), cv2.BORDER_DEFAULT
    )

    img_tensor = np.array(image)
    assert (
        temp_img.shape == img_tensor.shape
    ), f"Image dimensions do not match: {temp_img.shape} and {img_tensor.shape}!"
    assert type(temp_img) == type(
        img_tensor
    ), "Image and blurred image type does not match!"
    sum_mask_3d = np.dstack([bbl_mask] * 3)

    assert type(img_tensor) == type(sum_mask_3d), "Image and mask types do not match"
    assert (
        sum_mask_3d.shape == img_tensor.shape
    ), "Mask and image dimensions do not match!"

    img_tensor[sum_mask_3d == 0] = temp_img[sum_mask_3d == 0]
    # final = img_tensor * sum_mask_3d + temp_img * (1 - sum_mask_3d)

    return {"masked": img_tensor, "blurred": temp_img, "original": image}


if __name__ == "__main__":

    path = "imagenette2-320/train/n02102040/n02102040_8463.JPEG"

    ### DETR
    scores, boxes = main(path)
    im = Image.open(path)
    plot_box(im, scores, boxes, "OD-output.jpg")
    breakpoint()
    ## Blur
    a = background_blurrer(im, boxes)
    img = Image.fromarray(a["masked"], "RGB")
    img.save("masked.jpg")
    img_blur = Image.fromarray(a["blurred"], "RGB")
    img_blur.save("blurred.jpg")
