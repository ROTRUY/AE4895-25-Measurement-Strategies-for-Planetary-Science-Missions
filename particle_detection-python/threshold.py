"""
This code performs grain size distribution analysis and dumps results into a csv file.

Step 1: Read image and define pixel size (if needed to convert results into microns, not pixels)
Step 2: Denoising, if required and threshold image to separate grains from boundaries.
Step 3: Clean up image, if needed (erode, etc.) and create a mask for grains
Step 4: Label grains in the masked image
Step 5: Measure the properties of each grain (object)
Step 6: Output results into a csv file
"""
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color, measure

from helper import extract_and_save_region_properties

CSV_FOLDER = "./csv/threshold"
os.makedirs(CSV_FOLDER, exist_ok=True)
IMAGE_FOLDER = "./images"

def threshold_grain_detection(image_name: str, resolution: str, opening_radius = 50):
    # step 1
    img_gray = cv2.imread(f"{IMAGE_FOLDER}/{image_name}", 0)
    if resolution != "unknown":
        img_ref = cv2.imread(f"03022026/Quarz-Ref{resolution}.jpeg", 0)
        plt.imshow(img_ref)
        plt.title('Find pixel distance between references')
        plt.show()

    # step 2: denoising
    # Thresholding separates the grains from the boundaries
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(thresh)
    plt.title('Grains Thresholded')
    plt.show()

    # step 3: kernel
    # erosion & dilation enhance the grain boundaries
    kernel = np.ones((opening_radius, opening_radius), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(eroded)
    plt.title('Grains Eroded')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(dilated)
    plt.title('Grains Dilated')
    plt.axis('off')
    plt.show()

    # step 4 - label all the grains
    mask = dilated == 255
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=s)
    logging.info(f"Number of labels: {num_labels}")
    img2 = color.label2rgb(labeled_mask, bg_label=0)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.title('Grains Dilated')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Grains colored by label')
    plt.axis('off')
    plt.show()

    # step 5 - measure properties from all the grains
    regions = measure.regionprops(labeled_mask, intensity_image=img_gray)
    extract_and_save_region_properties(CSV_FOLDER, image_name, regions, resolution)

