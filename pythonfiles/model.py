import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
import numpy as np

def build_model(input_shape, num_layers,filters, kernel_size, dropout_rate = 0.1):
    # inp = layers.Input(shape=(None, *train_x.shape[2:], 1))
    # inp = layers.Input(shape=(None, *train_x.shape[2:]))
    # inp = layers.Input(shape=(train_x.shape[1:]))
    inp = layers.Input(shape=(input_shape))
    # inp = layers.Input(shape=(5, 64, 64))

    # out = keras.Sequential()
    # out = inp
    x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=True, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(inp)
    print('x after input layer: ' + str(x.shape))
    # x = layers.BatchNormalization()(x)
    for i in range(num_layers-2):
        x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=True, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(x)
        print('x in loop: ' + str(x.shape))
        # x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=False, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(x)
    # x = layers.BatchNormalization()(x)

    print('last convLSTM2D: ' + str(x.shape))

#     x = layers.Dense(64)(x)
    # x = layers.ConvLSTM2D(filters, kernel_size=(kernel, kernel), padding='same', return_sequences=False, activation='relu')(x)

    #try kernel size 1x1 instead of kernel_size, kernel_size, 1x1 used by https://arxiv.org/pdf/1506.04214.pdf
    x = layers.Conv2D(filters = 1,kernel_size=(1, 1), activation='relu', padding="same")(x)
    print('x at end: ' + str(x.shape))
    model = keras.models.Model(inp, x)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # model = Model(input= Input, output = layers)
    return model
