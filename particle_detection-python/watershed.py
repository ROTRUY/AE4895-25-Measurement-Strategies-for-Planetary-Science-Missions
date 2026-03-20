# https://www.youtube.com/watch?v=WyQ-3Fjay7A


"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

This code performs grain size distribution analysis and dumps results into a csv file.
It uses watershed segmentation for better segmentation.
Compare results to regular segmentation.
"""
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, measure
from skimage.segmentation import clear_border

from helper import extract_and_save_region_properties

CSV_FOLDER = "./csv/watershed"
os.makedirs(CSV_FOLDER, exist_ok=True)
IMAGE_FOLDER = "./particle_detection-python/images"


def watershed_grain_detection(image_name: str, resolution: str, opening_radius = 50) -> None:
    img_original = cv2.imread(f"{IMAGE_FOLDER}/{image_name}")
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove small noise - opening
    # To remove holes we can use closing
    kernel = np.ones((opening_radius, opening_radius), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


    # Check the total regions found before and after applying this.
    opening = clear_border(opening)  # Remove edge touching grains


    # Now we know that the regions at the center of cells is for sure cells
    # The region far away is background.
    # We need to extract sure regions. For that we can use erode.
    # But we have cells touching, so erode alone will not work.
    # To separate touching objects, the best approach would be distance transform and then thresholding.

    # let us start by identifying sure background area
    # dilating pixes a few times increases cell boundary to background.
    # This way whatever is remaining for sure will be background.
    # The area in between sure background and foreground is our ambiguous area.
    # Watershed should find this area for us.
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Finding sure foreground area using distance transform and thresholding
    # intensities of the points inside the foreground regions are changed to
    # distance their respective distances from the closest 0 value (boundary).
    # https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

    # Let us threshold the dist transform by 20% its max value.
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown ambiguous region is nothing but bkground - foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Now we create a marker and label the regions inside.
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0.
    # For markers let us use ConnectedComponents.
    ret3, markers = cv2.connectedComponents(sure_fg)

    # One problem rightnow is that the entire background pixels is given value 0.
    # This means watershed considers this region as unknown.
    # So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers + 10

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.

    # Now we are ready for watershed filling.
    markers = cv2.watershed(img_original, markers)
    # The boundary region will be marked -1
    # https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1


    # Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
    img_original[markers == -1] = [0, 255, 255]

    colored_processed = color.label2rgb(markers, bg_label=0)

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(img_original)
    plt.title("Overlay original image")
    plt.subplot(2, 2, 2)
    plt.imshow(colored_processed)
    plt.title('Colored grains')
    plt.subplot(2, 2, 3)
    plt.imshow(markers, cmap='jet')
    plt.title('markers')
    plt.subplot(2, 2, 4)
    plt.imshow(unknown, cmap='jet')
    plt.title('unknown')
    plt.show()

    # Now, time to extract properties of detected cells
    # regionprops function in skimage measure module calculates useful parameters for each object.
    regions = measure.regionprops(markers, intensity_image=img_gray)
    extract_and_save_region_properties(CSV_FOLDER, image_name, regions, resolution)
