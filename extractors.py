from typing import Union

import numpy as np
import matplotlib.image as mpimg

from features import hog_features, bin_spatial_features, color_hist_features
from utils import _convert_to_color


def _hog_features(
        img: np.array, hog_channel: Union[str, int], orient=9, pix_per_cell=8, cell_per_block=2
):
    """
    """
    if hog_channel == 'ALL':
        _features = []
        for channel in range(img.shape[2]):
            _features.append(
                hog_features(
                    img[:, :, channel],
                    orient, pix_per_cell, cell_per_block,
                    vis=False
                )
            )
            _features = np.ravel(_features)
    else:
        _features = hog_features(
            img[:, :, hog_channel], orient,
            pix_per_cell, cell_per_block, vis=False
        )
    return _features


def extract_features_for_image(
        img, cspace='RGB', spatial_size=(32, 32),
        hist_bins=32, orient=9,
        pix_per_cell=8, cell_per_block=2, hog_channel=0,
        spatial_feat=True, hist_feat=True, hog_feat=True
):
    """

    """
    # Create a list to append feature vectors to
    _features = []
    img_converted = _convert_to_color(img, cspace)

    # extract features
    if spatial_feat:
        spatial_features = bin_spatial_features(img_converted, spatial_size)
        _features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist_features(img_converted, nbins=hist_bins)
        _features.append(hist_features)

    if hog_feat:
        _features.append(
            _hog_features(
                img_converted,
                hog_channel,
                orient,
                pix_per_cell,
                cell_per_block
            )
        )
    return np.concatenate(_features)


def extract_features_for_images(
        images_paths, cspace, spatial_size, hist_bins, orient,
        pix_per_cell, cell_per_block, hog_channel,
        spatial_feat=True, hist_feat=True, hog_feat=True
):
    """
    """
    features = []

    for image_path in images_paths:
        img = mpimg.imread(image_path)
        img_flipped = np.fliplr(img)

        img_features = extract_features_for_image(
            img, cspace, spatial_size, hist_bins, orient, pix_per_cell,
            cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat
        )
        #
        # img_flipped_features = extract_features_for_image(
        #     img_flipped, cspace, spatial_size, hist_bins, orient, pix_per_cell,
        #     cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat
        # )

        features.append(img_features)
        #features.append(img_flipped_features)

    return features
