import os
import sys

import numpy as np
import pandas as pd
from random import randint, sample

import matplotlib.pyplot as plt
from tqdm import tqdm, tnrange

from sklearn.model_selection import train_test_split
from skimage.transform import resize

from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import tensorflow as tf

# path to image data
path_train = './data/train/'
path_test = './data/test/'

# resize image size for neural net
img_size_orig = 101
img_size_target = 128

def upsample(img):
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    return resize(img, (img_size_orig, img_size_orig), mode='constant', preserve_range=True)


def get_coverage_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
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


def main():
    # read train and test data index
    train_data = pd.read_csv('./train.csv', index_col='id', usecols=[0])
    depths_data = pd.read_csv('./depths.csv', index_col='id')
    train_data = train_data.join(depths_data)
    test_data = depths_data[~depths_data.index.isin(train_data.index)]

    # set up training data
    train_data['images'] = [np.array(load_img(
        path_train + 'images/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_data.index)]

    train_data['masks'] = [np.array(load_img(
        path_train + 'masks/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_data.index)]

    train_data['coverage'] = train_data.masks.map(np.sum) / pow(img_size_orig, 2)

    train_data['coverage_class'] = train_data.coverage.map(get_coverage_class)

    # split data into training and validation set
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_data.index.values,
        np.array(train_data.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_data.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_data.coverage.values,
        train_data.z.values,
        test_size=0.2, stratify=train_data.coverage_class, random_state=1337)

    # Build U-Net model
    inputs = Input((128, 128, 1))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30, 
                        callbacks=[earlystopper, checkpointer])

    pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm(enumerate(test_ids))}

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')


if __name__ == '__main__':
    main()
