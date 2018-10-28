#!/usr/bin/env python3

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply


# Unet model using csSE ans residual block
class UNetModel():
    def __init__(self, ckpt_name, loss, optimizer, metrics, monitor,
                 dropout_rate=0.25, epochs=50, batch_size=32, input_size=101,
                 input_layer=None, output_layer=None):
        if input_layer is None:
            self.input_layer = Input((input_size, input_size, 1))
        else:
            self.input_layer = input_layer
        if output_layer is None:
            self.output_layer = self.build_model(self.input_layer, 16, dropout_rate)
        else:
            self.output_layer = output_layer
        self.model = Model(self.input_layer, self.output_layer)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        self.model_name = ckpt_name
        self.model_checkpoint = ModelCheckpoint(self.model_name, monitor=monitor,
                                                mode='max', save_best_only=True, verbose=1)
        self.reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='max', factor=0.5,
                                           patience=5, min_lr=0.0001, verbose=1)

        self.epochs = epochs
        self.batch_size = batch_size

        print('''Model info:
            model name: {}
            loss: {}
            optimizer: {}
            monitor: {}
            dropout rate: {}
            epoch: {}
            batch size: {}'''.format(ckpt_name, loss, optimizer, monitor, dropout_rate, epochs, batch_size))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.history = self.model.fit(x_train, y_train,
                                      validation_data=[x_valid, y_valid],
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      callbacks=[self.model_checkpoint, self.reduce_lr],
                                      verbose=2)
        return self.history

    def batch_activate(self, x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def convolution_block(self, x, filters, size, strides=(1, 1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation is True:
            x = self.batch_activate(x)
        return x

    def residual_block(self, blockInput, num_filters=16, activation=False):
        x = self.batch_activate(blockInput)
        x = self.convolution_block(x, num_filters, (3, 3))
        x = self.convolution_block(x, num_filters, (3, 3), activation=False)
        x = Add()([x, blockInput])
        if activation:
            x = self.batch_activate(x)
        return x

    def squeeze_excite_block_cSE(self, input, ratio=2):
        init = input

        filters = K.int_shape(init)[-1]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)

        x = multiply([init, se])
        return x

    def squeeze_excite_block_sSE(sekf, input):
        sSE_scale = Conv2D(1, (1, 1), activation='sigmoid', padding="same", use_bias=True)(input)
        return multiply([input, sSE_scale])

    def unet_layer(self, blockInput, num_filters, use_csSE_ratio=2):
        x = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
        x = self.residual_block(x, num_filters)
        x = self.residual_block(x, num_filters, activation=True)

        if use_csSE_ratio > 0:
            sSEx = self.squeeze_excite_block_sSE(x)
            cSEx = self.squeeze_excite_block_cSE(x, ratio=use_csSE_ratio)
            x = Add()([sSEx, cSEx])

        return x

    def build_model(self, input_layer, start_neurons, DropoutRatio=0.5, use_csSE_ratio=2):
        # 101 -> 50
        conv1 = self.unet_layer(input_layer, start_neurons * 1, use_csSE_ratio)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(DropoutRatio / 2)(pool1)

        # 50 -> 25
        conv2 = self.unet_layer(pool1, start_neurons * 2, use_csSE_ratio)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(DropoutRatio)(pool2)

        # 25 -> 12
        conv3 = self.unet_layer(pool2, start_neurons * 4, use_csSE_ratio)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(DropoutRatio)(pool3)

        # 12 -> 6
        conv4 = self.unet_layer(pool3, start_neurons * 8, use_csSE_ratio)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(DropoutRatio)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
        convm = self.residual_block(convm, start_neurons * 16)
        convm = self.residual_block(convm, start_neurons * 16, True)

        # 6 -> 12
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(DropoutRatio)(uconv4)

        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = self.residual_block(uconv4, start_neurons * 8)
        uconv4 = self.residual_block(uconv4, start_neurons * 8, True)

        # 12 -> 25
        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(DropoutRatio)(uconv3)

        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = self.residual_block(uconv3, start_neurons * 4)
        uconv3 = self.residual_block(uconv3, start_neurons * 4, True)

        # 25 -> 50
        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])

        uconv2 = Dropout(DropoutRatio)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = self.residual_block(uconv2, start_neurons * 2)
        uconv2 = self.residual_block(uconv2, start_neurons * 2, True)

        # 50 -> 101
        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
        uconv1 = concatenate([deconv1, conv1])

        uconv1 = Dropout(DropoutRatio)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = self.residual_block(uconv1, start_neurons * 1)
        uconv1 = self.residual_block(uconv1, start_neurons * 1, True)

        output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
        output_layer = Activation('sigmoid')(output_layer_noActi)

        return output_layer
