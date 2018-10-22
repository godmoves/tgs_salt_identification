import os
import pprint
import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

from sklearn.model_selection import train_test_split
from skimage.transform import resize

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply

from tqdm import tqdm

# model base name
MODEL_TYPE = 'res_Unet_csSE_'
MODEL_VERSION = 'V5'
MODEL_BASE_NAME = MODEL_TYPE + MODEL_VERSION

# resize image size for neural net
IMG_SIZE_ORIGINGIN = 101
IMG_SIZE_TARGET = 101

def upsample(img):
    if IMG_SIZE_ORIGIN == IMG_SIZE_TARGET:
        return img
    return resize(img, (IMG_SIZE_TARGET, IMG_SIZE_TARGET), mode='constant', preserve_range=True)


def downsample(img):
    if IMG_SIZE_ORIGIN == IMG_SIZE_TARGET:
        return img
    return resize(img, (IMG_SIZE_ORIGIN, IMG_SIZE_ORIGIN), mode='constant', preserve_range=True)


def get_coverage_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


def predict_result(model,x_test,IMG_SIZE_TARGET): 
    # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test)
    preds_test2_refect = model.predict(x_test_reflect)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
    return preds_test / 2


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


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 ) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)


# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -infinity and +infinity)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        # elu doesn't perform better
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss


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


def batch_activate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = batch_activate(x)
    return x

def residual_block(blockInput, num_filters=16, activation = False):
    x = batch_activate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if activation:
        x = batch_activate(x)
    return x


def squeeze_excite_block_cSE(input, ratio=2):
    init = input

    filters = K.int_shape(init)[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)

    x = multiply([init, se])
    return x


def squeeze_excite_block_sSE(input):
    sSE_scale = Conv2D(1, (1, 1), activation='sigmoid', padding="same", use_bias = True)(input)
    return multiply([input, sSE_scale])


def unet_layer(blockInput, num_filters, use_csSE_ratio = 2):
    x = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
    x = residual_block(x, num_filters )
    x = residual_block(x, num_filters , activation = True)

    if use_csSE_ratio > 0:
        sSEx = squeeze_excite_block_sSE(x)
        cSEx = squeeze_excite_block_cSE(x,ratio = use_csSE_ratio ) # modified 10/10/2018
        x = Add()([sSEx, cSEx])

    return x


# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5, use_csSE_ratio=2):
    # 101 -> 50
    conv1 = unet_layer(input_layer, start_neurons * 1, use_csSE_ratio)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = unet_layer(pool1, start_neurons * 2, use_csSE_ratio)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = unet_layer(pool2, start_neurons * 4, use_csSE_ratio)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = unet_layer(pool3, start_neurons * 8, use_csSE_ratio)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer


def main():
    # Data preprocessing
    # read train and test data index
    train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("./data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    # set up training data
    train_data['images'] = [np.array(load_img(
        path_train + './data/train/images/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_data.index)]

    train_data['masks'] = [np.array(load_img(
        path_train + './data/train/masks/{}.png'.format(idx), grayscale=True)) / 255 for idx in tqdm(train_data.index)]

    train_data['coverage'] = train_data.masks.map(np.sum) / pow(IMG_SIZE_ORIGINGIN, 2)

    train_data['coverage_class'] = train_data.coverage.map(get_coverage_class)

    # we exclude all data whose coverage is less than 1.5% but not fully empty
    # this method can improve the training accuracy, but may not be good in all conditions
    train_df_notempty = train_df[(train_df.coverage > 0.015) | (train_df.coverage == 0)]
    print('remain data number:', len(train_df_notempty))


    # Split data into train and test sets stratified by salt coverage
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_valid, depth_train, depth_valid = train_test_split(
        train_df_notempty.index.values,
        np.array(train_df_notempty.images.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1), 
        np.array(train_df_notempty.masks.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1), 
        train_df_notempty.coverage.values,
        train_df_notempty.z.values,
        test_size=0.2, stratify=train_df_notempty.coverage_class, random_state=1337)

        # oooops, do not forget data augmentation
    # we augment the data by fliping left and right
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    # Build U-Net model
    # first stage
    input_layer = Input((IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1))
    output_layer = build_model(input_layer, 32, 0.25)
    model_pre = Model(input_layer, output_layer)
    model_pre.compile(loss="binary_crossentropy", optimizer='adam', metrics=[my_iou_metric])

    model_pre_name = MODEL_BASE_NAME + '_pre.model'

    model_checkpoint = ModelCheckpoint(model_pre_name, monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5,
                                  patience=5, min_lr=0.0001, verbose=1)

    epochs = 100
    batch_size = 32
    history = model_pre.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid], 
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint, reduce_lr], 
                            verbose=2)

    # second stage using lovasz loss
    # Split data into train and test sets stratified by salt coverage
    # this time we add all data back
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_valid, depth_train, depth_valid = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1), 
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1), 
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    # data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    model_pre = load_model(model_pre_name, custom_objects={'my_iou_metric': my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model_pre.layers[0].input
    output_layer = model_pre.layers[-1].input

    # build the final predict model
    model = Model(input_x, output_layer)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer='adam', metrics=[my_iou_metric_2])

    model_name = MODEL_BASE_NAME + '.model'

    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode='max',
                                   patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric_2', 
                                       mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode='max',
                                  factor=0.5, patience=5, min_lr=0.0001, verbose=1)

    epochs = 120
    batch_size = 32
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint,reduce_lr,early_stopping], 
                        verbose=2)


    # restore the best model
    model = load_model(model_name, custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})

    preds_valid = predict_result(model, x_valid, IMG_SIZE_TARGET)

    # Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    # TODO: there are two ways of scoring the result. test them later
    ious = np.array(
        [get_iou_vector(y_valid, preds_valid.reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1) > threshold) for threshold in tqdm(thresholds)])
    print(ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print('threshold_best', threshold_best)

    # predict the result on test set
    x_test = np.array([upsample(np.array(load_img(
        "./data/test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)

    preds_test = model.predict(x_test)
    # test time augmention
    x_test2 = np.array([np.fliplr(x) for x in x_test])
    preds_test2 = model.predict(x_test2)
    preds_test2 = np.array([np.fliplr(x) for x in preds_test2])
    preds_test = (preds_test + preds_test2) / 2

    pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')


if __name__ == '__main__':
    main()
