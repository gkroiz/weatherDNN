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
from skimage.io import imread
from skimage.transform import resize
# import math
from csv import reader
# import random

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def plotTraining(history):
    print('in plotting function')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('/home/gkroiz1/weatherDNN/pythonfiles/trainingplot.pdf')

#class to load in data that does not fit into memory
class customDataLoader(keras.utils.Sequence):
    def __init__(self, tileIDs, batch_size, steps_per_epoch, lead_frames_x, lead_frames_y, data_loc, dataType = 'err'):

        #check to make sure you are working with train, val, or test data
        dataTypes = ['train','val','test']
        if dataType not in dataTypes:
            raise ValueError('Invalid data type for custom data loader. Expected one of %s' %dataTypes)

        #define class variables
        self.x_data, self.y_data = np.array([-1]), np.array([-1])
        self.batch_size = batch_size
        self.all_tiles = tileIDs
        self.steps_per_epoch = steps_per_epoch
        self.data_loc = data_loc
        self.lead_frames_x = lead_frames_x
        self.lead_frames_y = lead_frames_y
        self.dataType = dataType
        self.batches_in_tile_counter = 0
        self.tile_counter = 0
        self.on_epoch_end()


    #returns number of steps per epoch, based on calculation outside of function
    def __len__(self):
        return self.steps_per_epoch - 1

    #returns one batch
    # def __getitem__(self):
    def __getitem__(self, index):
        if (self.x_data.shape == (1,)):
            data = np.load(self.data_loc + self.dataType + '-t-' + str(self.all_tiles[self.tile_counter]) + '.npy')
            self.x_data = data[:,0:lead_frames_x]
            self.y_data = data[:,lead_frames_x-1:lead_frames_x+lead_frames_y-1]

        self.batches_in_tile_counter += 1
        if (self.batches_in_tile_counter) * self.batch_size-1 > self.x_data.shape[0]:
            x = np.array(self.x_data[(self.batches_in_tile_counter-1) * self.batch_size:-1])
            y = np.array(self.y_data[(self.batches_in_tile_counter-1) * self.batch_size:-1])
            self.x_data = np.array([-1])
            self.y_data = np.array([-1])
            self.batches_in_tile_counter = 0
            self.tile_counter += 1
            return x, y
        else:
            x = np.array(self.x_data[(self.batches_in_tile_counter-1) * self.batch_size:self.batches_in_tile_counter * self.batch_size-1])
            y = np.array(self.y_data[(self.batches_in_tile_counter-1) * self.batch_size:self.batches_in_tile_counter * self.batch_size-1])

            return x, y

    def on_epoch_end(self):
        print("HERE")
        # self.tiles_left = self.all_tiles
        self.batches_in_tile_counter = 0
        self.tile_counter = 0

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)

    print('tf version:' + str(tf.__version__))
    batch_size = 32
    num_tiles = 5#75
    epochs = 5
    train_loc = json_params["train_loc"]
    val_loc = json_params["val_loc"]
    test_loc = json_params["test_loc"]
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]
    data_info_loc = json_params["data_info_loc"]
    lead_frames_x = int(lead_time_x/5)
    lead_frames_y = int(lead_time_y/5)

    train_data, val_data, test_data = [], [], []

    # needMakeData = False
    # if os.path.isdir(train_loc):
    #     train_data = np.load(train_loc)
    # else:
    #     needMakeData = True
    # if os.path.isfile(val_loc):
    #    val_data = np.load(val_loc)
    # else:
    #     needMakeData = True
    # if os.path.isfile(test_loc):
    #     test_data = np.load(test_loc)
    # else:
    #     needMakeData = True


    
    # # exit()
    # print('needMakeData: ' + str(needMakeData))
    # # randomlyGenTiles = [218, 103, 82, 57, 14, 24, 198, 163, 110, 66, 93, 133, 15, 
    # # 51, 115, 151, 37, 234, 220, 85, 26, 56, 84, 183, 119, 105, 9, 137, 205, 
    # # 112, 250, 200, 131, 75, 177, 213, 207, 148, 178, 221, 125, 67, 169, 60, 
    # # 7, 204, 224, 228, 40, 181, 171, 255, 231, 100, 86, 74, 55, 114, 104, 180, 
    # # 168, 145, 238, 79, 187, 45, 116, 4, 245, 88, 147, 155, 23, 252, 18]
    # # randomlyGenTiles = [218]
    # if needMakeData:
    #     # tilesIDs = list(range(256))
    #     tilesIDs = np.random.randint(0,256, 75)
    #     for i in range(len(tilesIDs)):
    #         # singleTile = random.choice(tilesIDs)
    #         train_val_test_gen(train_loc, val_loc, test_loc, data_info_loc, [tilesIDs[i]])
    #         # tilesIDs.remove(singleTile)
    
    
    tileIDs = []

    data_info_df = pd.read_csv(data_info_loc, sep = ',', header = 0)
    
    tileIDs = data_info_df['tileID']
    print(tileIDs)
    # exit()
    train_steps_per_epoch = int(np.sum((data_info_df['train_num_samples']/batch_size).apply(np.ceil)))
    val_steps_per_epoch = int(np.sum((data_info_df['val_num_samples']/batch_size).apply(np.ceil)))



    # print('lead_frames_x: ' + str(lead_frames_x))
    # print('lead_frames_y: ' + str(lead_frames_y))
    # train_x = train_data[:,0:lead_frames_x]
    # train_y = train_data[:,lead_frames_x-1:lead_frames_x+lead_frames_y-1]

    # val_x = val_data[:,0:lead_frames_x]
    # val_y = val_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]

    # test_x = test_data[:,0:lead_frames_x]
    # test_y = test_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]
    print('train steps: ' + str(train_steps_per_epoch) + ', val steps:' + str(val_steps_per_epoch))

    train_batch_generator = customDataLoader(tileIDs, batch_size, train_steps_per_epoch, lead_frames_x, lead_frames_y, train_loc, 'train')
    val_batch_generator = customDataLoader(tileIDs, batch_size, val_steps_per_epoch, lead_frames_x, lead_frames_y, val_loc, 'val')

    print('before reshaping')
    # train_x = train_x.reshape((train_x.shape[0], lead_time_x, 64, 64))
    # train_y = train_y.reshape((train_y.shape[0], lead_time_y))

    #kernel = 3 based on https://arxiv.org/pdf/1506.04214.pdf, kernel 5 results in more complex model

    input_shape = (None, 64, 64, 1)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(input_shape, num_layers = 2, filters = 64, kernel_size = 3)
    opt = keras.optimizers.Adam(learning_rate=1e-2)



    model.compile(loss='mse', optimizer=opt, metrics =['mse', 'accuracy'])
    
    early_stopping = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1, 
                        mode='auto'
                    )
    learning_rate = keras.callbacks.LearningRateScheduler(scheduler)

    print(model.summary())

    # history = model.fit(x = train_batch_generator, steps_per_epoch = train_steps_per_epoch, validation_data = val_batch_generator, validation_steps = val_steps_per_epoch, epochs=epochs, verbose = 2, callbacks=[OnEpochEnd([train_batch_generator.on_epoch_end]), early_stopping, learning_rate])
    history = model.fit(x = train_batch_generator, validation_data = val_batch_generator, epochs=epochs, verbose = 2, callbacks=[early_stopping, learning_rate])

    model.save('trained_model.h5')

    #load model
    # keras.models.load_model('trained_model.h5')


    # plotTraining(history)

    print('after plotting')
    #write code to test model: