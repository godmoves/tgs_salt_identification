#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tqdm import tqdm

from helper import predict_result, RLenc
from helper import upsample, downsample, get_coverage_class
from loss import lovasz_loss
from metric import my_iou_metric, my_iou_metric_2, get_iou_vector
from model import UNetModel

# Model base name
MODEL_TYPE = 'res_Unet_csSE_'
MODEL_VERSION = 'V5'
MODEL_BASE_NAME = MODEL_TYPE + MODEL_VERSION

# Resize image size for neural net
IMG_SIZE_ORIGIN = 101
IMG_SIZE_TARGET = 101


def main():
    # Data preprocessing
    # Read train and test data index
    train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("./data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    # Set up training data
    train_df['images'] = [np.array(load_img(
        './data/train/images/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    train_df['masks'] = [np.array(load_img(
        './data/train/masks/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    train_df['coverage'] = train_df.masks.map(np.sum) / pow(IMG_SIZE_ORIGIN, 2)

    train_df['coverage_class'] = train_df.coverage.map(get_coverage_class)

    # We exclude all data whose coverage is less than 1.5% but not fully empty
    # this method can improve the training accuracy, but may not be good in all conditions
    train_df_notempty = train_df[(train_df.coverage > 0.015) | (train_df.coverage == 0)]
    print('remain data number:', len(train_df_notempty))

    # Split data into train and test sets stratified by salt coverage
    x_train, x_valid, y_train, y_valid = train_test_split(
        np.array(train_df_notempty.images.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1),
        np.array(train_df_notempty.masks.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1),
        test_size=0.2, stratify=train_df_notempty.coverage_class, random_state=1337)

    # Augment the data by fliping left and right
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    # Build U-Net model
    # First training stage with binary loss
    print('Stage 1, using binary loss')

    model_pre_name = MODEL_BASE_NAME + '_pre.model'

    model_pre = UNetModel(ckpt_name=model_pre_name,
                          loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[my_iou_metric],
                          monitor='val_my_iou_metric',
                          dropout_rate=0.25,
                          epochs=50,
                          batch_size=32)

    model_pre.fit(x_train, y_train, x_valid, y_valid)

    # Second stage using lovasz loss
    # Split data into train and test sets stratified by salt coverage
    # Ihis time we add all data back
    x_train, x_valid, y_train, y_valid = train_test_split(
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1),
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1),
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    # Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    print('Stage 2, using lovasz loss')
    model_pre = load_model(model_pre_name, custom_objects={'my_iou_metric': my_iou_metric})
    # Remove layter activation layer and use lovasz loss
    model_name = MODEL_BASE_NAME + '.model'
    model = UNetModel(ckpt_name=model_name,
                      loss=lovasz_loss,
                      optimizer='adam',
                      metrics=[my_iou_metric_2],
                      monitor='val_my_iou_metric_2',
                      epochs=50,
                      batch_size=32,
                      input_layer=model_pre.layers[0].input,
                      output_layer=model_pre.layers[-1].input)

    model.fit(x_train, y_train, x_valid, y_valid)

    # Restore the best model
    model = load_model(model_name, custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})

    preds_valid = predict_result(model, x_valid)

    # Scoring for last model, choose threshold by validation data
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori / (1 - thresholds_ori))

    ious = np.array([get_iou_vector(y_valid, preds_valid.reshape(
        -1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1) > threshold) for threshold in tqdm(thresholds)])
    print('\nFinding best threshold...\n', ious)

    # Instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print('Threshold_best', threshold_best)
    print('IoU_best', iou_best)

    # Predict the result on test set
    x_test = np.array([upsample(np.array(load_img(
        "./data/test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(
            -1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)

    preds_test = model.predict(x_test)

    # Test time augmention by fliping left and right
    x_test2 = np.array([np.fliplr(x) for x in x_test])
    preds_test2 = model.predict(x_test2)
    preds_test2 = np.array([np.fliplr(x) for x in preds_test2])
    preds_test = (preds_test + preds_test2) / 2

    pred_dict = {idx: RLenc(np.round(downsample(
        preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')


if __name__ == '__main__':
    main()
