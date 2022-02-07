#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 00:09:24 2020

@author: avanetten
Update to:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/_skeletonize.py#L364
"""

import sys

# add path of scikit-image
# skimage_path = '/Users/avanetten/opt/anaconda3/envs/solaris/lib/python3.6/site-packages/skimage/morphology'
# sys.path.append(skimage_path)

import numpy as np
from scipy import ndimage as ndi
import skimage
# from skimage.morphology import *
from skimage.morphology._skeletonize import _table_lookup, _pattern_of
from skimage.morphology._skeletonize_cy import (_skeletonize_loop,
                                                _table_lookup_index)


_eight_connect = ndi.generate_binary_structure(2, 2)


###############################################################################
def medial_axis_weight(image, weight_arr=None, mask=None,
                       return_distance=False,
                       verbose=False):
    """
    Compute the medial axis transform of a binary image, use weight_arr if
    desired
    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to be skeletonized.
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.
    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)
    See also
    --------
    skeletonize
    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.
    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.
     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.
     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.
     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.
    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """

    if verbose:
        print("medial_axis_transform: image.shape:", image.shape)
        if weight_arr is not None:
            print("medial_axis_transform: weight_arr.shape:", weight_arr.shape)

    global _eight_connect
    if mask is None:
        masked_image = image.astype(np.bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Build lookup table - three conditions
    # 1. Keep only positive pixels (center_is_foreground array).
    # AND
    # 2. Keep if removing the pixel results in a different connectivity
    # (if the number of connected components is different with and
    # without the central pixel)
    # OR
    # 3. Keep if # pixels in neighbourhood is 2 or less
    # Note that table is independent of image
    center_is_foreground = (np.arange(512) & 2**4).astype(bool)
    table = (center_is_foreground  # condition 1.
                &
            (np.array([ndi.label(_pattern_of(index), _eight_connect)[1] !=
                       ndi.label(_pattern_of(index & ~ 2**4),
                                    _eight_connect)[1]
                       for index in range(512)])  # condition 2
                |
        np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
        # condition 3
            )

    # Build distance transform
    distance = ndi.distance_transform_edt(masked_image)
    if verbose:
        print("medial_axis_transform: init distance.shape:", distance.shape)
    if return_distance:
        store_distance = distance.copy()
    
    if weight_arr is not None:
        distance = distance * weight_arr

    # Corners
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    # We use a cornerness_table lookup table where the score of a
    # configuration is the number of background (0-value) pixels in the
    # 3x3 neighbourhood
    cornerness_table = np.array([9 - np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)

    # Define arrays for inner loop
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]

    if verbose:
        print("medial_axis_transform: stage2 distance.shape:", distance.shape)

    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)

    # Determine the order in which pixels are processed.
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    generator = np.random.RandomState(0)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker,
                        corner_score[masked_image],
                        distance))
    if verbose:
        print("medial_axis_transform: distance.shape:", distance.shape)
        print("medial_axis_transform: corner_score[masked_image].shape:",
              corner_score[masked_image].shape)
        print("medial_axis_transform: tiebreakder.shape:",
              tiebreaker.shape)

    order = np.ascontiguousarray(order, dtype=np.int32)

    table = np.ascontiguousarray(table, dtype=np.uint8)
    # Remove pixels not belonging to the medial axis
    _skeletonize_loop(result, i, j, order, table)

    result = result.astype(bool)
    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return result, store_distance
    else:
        return result


###############################################################################
def medial_axis_sort(image, seed_arr=None, mask=None, return_distance=False,
                     verbose=False):
    """
    Compute the medial axis transform of a binary image, use seed_arr if
    desired
    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to be skeletonized.
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.
    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)
    See also
    --------
    skeletonize
    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.
    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.
     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.
     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.
     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.
    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    
    if verbose:
        print("medial_axis_transform: image.shape:", image.shape)
        if seed_arr is not None:
            print("medial_axis_transform: seed_arr.shape:", seed_arr.shape)

    global _eight_connect
    if mask is None:
        masked_image = image.astype(np.bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Build lookup table - three conditions
    # 1. Keep only positive pixels (center_is_foreground array).
    # AND
    # 2. Keep if removing the pixel results in a different connectivity
    # (if the number of connected components is different with and
    # without the central pixel)
    # OR
    # 3. Keep if # pixels in neighbourhood is 2 or less
    # Note that table is independent of image
    center_is_foreground = (np.arange(512) & 2**4).astype(bool)
    table = (center_is_foreground  # condition 1.
                &
            (np.array([ndi.label(_pattern_of(index), _eight_connect)[1] !=
                       ndi.label(_pattern_of(index & ~ 2**4),
                                    _eight_connect)[1]
                       for index in range(512)])  # condition 2
                |
        np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
        # condition 3
            )

    # Build distance transform
    distance = ndi.distance_transform_edt(masked_image)
    if verbose:
        print("medial_axis_transform: init distance.shape:", distance.shape)
    if return_distance:
        store_distance = distance.copy()

    # Corners
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    # We use a cornerness_table lookup table where the score of a
    # configuration is the number of background (0-value) pixels in the
    # 3x3 neighbourhood
    cornerness_table = np.array([9 - np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)

    # Define arrays for inner loop
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]

    if verbose:
        print("medial_axis_transform: stage2 distance.shape:", distance.shape)

    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)

    # Determine the order in which pixels are processed.
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    generator = np.random.RandomState(0)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    if seed_arr is None:
        order = np.lexsort((tiebreaker,
                            corner_score[masked_image],
                            distance))
    else:
        # use the seed_arr as the first determinant of order
        # mask seed_arr
        mask_tmp = masked_image.copy()
        seed_arr = seed_arr[mask_tmp]

        if verbose:
            print("medial_axis_transform: seed_arr.shape:", seed_arr.shape)
            print("medial_axis_transform: distance.shape:", distance.shape)
            print("medial_axis_transform: corner_score[masked_image].shape:",
                  corner_score[masked_image].shape)
            print("medial_axis_transform: tiebreakder.shape:",
                  tiebreaker.shape)
        
        order = np.lexsort((tiebreaker,
                            corner_score[masked_image],
                            distance,
                            seed_arr))

    order = np.ascontiguousarray(order, dtype=np.int32)

    table = np.ascontiguousarray(table, dtype=np.uint8)
    # Remove pixels not belonging to the medial axis
    _skeletonize_loop(result, i, j, order, table)

    result = result.astype(bool)
    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return result, store_distance
    else:
        return result
