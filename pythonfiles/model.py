# from re import I
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
import numpy as np

def build_model(train_x, num_layers,filters, kernel):
    # inp = layers.Input(shape=(None, *train_x.shape[2:], 1))
    inp = layers.Input(shape=(None, *train_x.shape[2:]))

    # inp = layers.Input(shape=(5, 64, 64))

    # out = keras.Sequential()
    # out = inp
    x = layers.ConvLSTM2D(filters, kernel_size=(kernel, kernel), padding='same', return_sequences=True, activation='relu')(inp)
    for i in range(num_layers-1):
        print('in for loop')
        x = layers.ConvLSTM2D(filters, kernel_size=(kernel, kernel), padding='same', return_sequences=True, activation='relu')(x)
#     x = layers.Dense(64)(x)
    x = layers.Conv2D(
        filters=1, kernel_size=(kernel, kernel), activation="sigmoid", padding="same")(x)
    
    model = keras.models.Model(inp, x)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # model = Model(input= Input, output = layers)
    return model
