from queue import Queue

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def _gray_to_RGB(gray):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img


def _binary_to_RGB(binary):
    return np.dstack((binary, binary, binary)) * 255


# solution from https://stackoverflow.com/a/47334314
def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        if column == no_of_columns + 1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        plt.imshow(list_of_images[i])
        plt.axis('off')
        if len(list_of_titles) > i:
            plt.title(list_of_titles[i])


def convert_color(img, conv='RGB2YCrCb'):
    """
    Converts the images' colors based on the full openCV conversion name
    :param img:
    :param conv:
    :return:
    """
    return cv2.cvtColor(img, getattr(cv2, 'COLOR_{}'.format(conv)))


def _convert_to_color(img, cspace='RGB'):
    """
        Converts image to cspace color space
    """
    if cspace != 'RGB':
        img_converted = convert_color(img, 'RGB2{}'.format(cspace))
    else:
        img_converted = np.copy(img)
    return img_converted


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    img_copy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img_copy
