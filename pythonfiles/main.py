# file description: where main is located, generates data, compiles model,
# trains model, and eval's model

import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4
import xarray as xr
import numpy as np
import pickle
from model import build_model
from preprocessing import train_val_test_gen
from matplotlib import pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

def plotTraining(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('trainingplot.pdf')

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)

    train_loc = json_params["train_loc"]
    val_loc = json_params["val_loc"]
    test_loc = json_params["test_loc"]
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]
    lead_frames_x = int(lead_time_x/5)
    lead_frames_y = int(lead_time_y/5)

    train_data, val_data, test_data = [], [], []

    needMakeData = False
    if os.path.isfile(train_loc):
        train_data = np.load(train_loc)
    else:
        needMakeData = True
    if os.path.isfile(val_loc):
       val_data = np.load(val_loc)
    else:
        needMakeData = True
    if os.path.isfile(test_loc):
        test_data = np.load(test_loc)
    else:
        needMakeData = True

    if needMakeData:
        train_val_test_gen(train_loc, val_loc, test_loc)
    
    print('lead_frames_x: ' + str(lead_frames_x))
    print('lead_frames_y: ' + str(lead_frames_y))
    train_x = train_data[:,0:lead_frames_x]
    train_y = train_data[:,lead_frames_x-1:lead_frames_x+lead_frames_y-1]

    val_x = val_data[:,0:lead_frames_x]
    val_y = val_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]

    test_x = test_data[:,0:lead_frames_x]
    test_y = test_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]

    print('before reshaping')
    # train_x = train_x.reshape((train_x.shape[0], lead_time_x, 64, 64))
    # train_y = train_y.reshape((train_y.shape[0], lead_time_y))

    #kernel = 3 based on https://arxiv.org/pdf/1506.04214.pdf, kernel 5 results in more complex model
    model = build_model(train_x, num_layers = 2, filters = 64, kernel_size = 3)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=opt, metrics =['mse', 'accuracy'])

    print(model.summary())
    history = model.fit(train_x, train_y, batch_size=64, validation_data=(val_x, val_y), epochs=50, verbose = 2,
        callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=5,
                        verbose=1, 
                        mode='auto'
                    )])

    model.save('trained_model.h5')

    #load model
    # keras.models.load_model('trained_model.h5')


    plotTraining(history)

    #write code to test model: