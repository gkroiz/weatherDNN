#################################################################################
# file name: model.py                                                           #
# author: Gerson Kroiz                                                          #
# file desc: this model includes the model architecture using tensorflow API    #
#################################################################################

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#################################################################################
# function: build_model                                                         #
# description: creates model                                                    #
# inputs:                                                                       #
# 1) input_shape: shape of input data                                           #
# 2) num_layers: minimum number of convLSTM blocks is 2,                        #
#    this is the total number of convLSTM blocks - 2                            #
# 3) filters: number of filters for each convolutional layer                    #
# 4) kernel_size: size of kernel for convolutional layers                       #
# 5) dropout_rate: dropout rate for hidden and input states, default is 0.1     #
# outputs:                                                                      #
# 1) model: the full model                                                      #
################################################################################# 
def build_model(input_shape, num_layers,filters, kernel_size, dropout_rate = 0.1):
    inp = layers.Input(shape=(input_shape))

    x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=True, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(inp)
    print('x after input layer: ' + str(x.shape))
    for i in range(num_layers-2):
        x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=True, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(x)
        print('x in loop: ' + str(x.shape))

    x = layers.ConvLSTM2D(filters, (kernel_size, kernel_size), padding='same', data_format='channels_last', return_sequences=False, activation='relu', dropout = dropout_rate, recurrent_dropout = dropout_rate)(x)

    print('last convLSTM2D: ' + str(x.shape))


    x = layers.Conv2D(filters = 1,kernel_size=(1, 1), activation='relu', padding="same")(x)
    print('x at end: ' + str(x.shape))
    model = keras.models.Model(inp, x)

    return model
