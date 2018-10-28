#!/usr/bin/env python3

import numpy as np
from skimage.transform import resize


# This time we don't use up/down sample
def upsample(img, origin=101, target=101):
    if origin == target:
        return img
    return resize(img, (target, target), mode='constant', preserve_range=True)


def downsample(img, origin=101, target=101):
    if origin == target:
        return img
    return resize(img, (origin, origin), mode='constant', preserve_range=True)


def get_coverage_class(val):
    # Split data into 10 classes according to salt coverage
    for i in range(0, 11):
        if val * 10 <= i:
            return i


def predict_result(model, x_test):
    # predict both orginal and reflect x
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test)
    preds_test2_refect = model.predict(x_test_reflect)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
    return preds_test / 2


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
