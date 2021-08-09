#################################################################################
# file name: main.py                                                            #
# author: Gerson Kroiz                                                          #
# file desc: this file takes the convLSTM from                                  #
# model.py, creates training and validation datasets based on npy files,        #
# and trains the model on the datasets using Tensorflow's                       #
# distributed training strategies                                               #
# requirements: main.json, preprocessing.py, model.py                           #
#################################################################################

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

from csv import reader
from sys import getsizeof

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

#for plotting
from matplotlib import pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from csv import reader
from sys import getsizeof

#################################################################################
# function: scheduler                                                           #
# description: decay loss function                                              #
# inputs:                                                                       #
# 1) epoch: what epoch the model is on                                          #
# 2) lr: the starting learning rate                                             #
# outputs:                                                                      #
# 1) value resulting from the scheduler                                         #
################################################################################# 
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

#################################################################################
# function: plotTraining                                                        #
# description: plots the validation loss and loss over epoch                    #
# inputs:                                                                       #
# 1) history: training history from main                                        #
################################################################################# 
def plotTraining(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('/home/gkroiz1/weatherDNN/pythonfiles/trainingval.pdf')


#################################################################################
# class: customDataLoader                                                       #
# description: based on keras.utils.Sequence, it is a data loader that only     #
#              loads in a batch worth of data into memory. The dataLoader loads #
#              data in sequential order based on the ordering of files provided #
#              in meta_data.csv                                                 #
################################################################################# 
class customDataLoader(keras.utils.Sequence):

    #################################################################################
    # function: __init__                                                            #
    # description: constructor for the customDataLoader                             #
    # inputs:                                                                       #
    # 1) self: class instance                                                       #
    # 2) tileIDs: which tiles are included in the entire dataset, is an array       #
    # 3) num_samples_per_tile: number of samples that are within each tile provided #
    #    by metadata meta_data.csv                                                  #
    # 4) batch_size: size of a batch                                                #
    # 5) steps_per_epoch: ceil(number of total samples in dataset / batch_size)     #
    # 6) lead_frames_x: number of consecutive time frames in x_data                 #        
    # 7) lead_frames_y: number of consecutive time frames in y_data                 #
    # 8) tileSize: size of the tiles                                                #
    # 9) data_loc: location of the data                                             #
    # 10) dataType: default is 'err', possible values are 'train', 'val', and       #
    #     'test'. This helps distinguish what type of data the dataLoader loads     #
    #     whether it is train, val, or test data                                    #
    ################################################################################# 
    def __init__(self, tileIDs, num_samples_per_tile, batch_size, steps_per_epoch, 
    lead_frames_x, lead_frames_y, tileSize, data_loc, dataType = 'err'):

        #check to make sure you are working with train, val, or test data
        dataTypes = ['train','val','test']
        if dataType not in dataTypes:
            raise ValueError('Invalid data type for custom data loader. Expected one of %s' %dataTypes)

        #define class variables

        #set data as an empty array
        self.data = np.array([-1])

        self.batch_size = batch_size
        self.all_tiles = tileIDs
        self.num_samples_per_tile =  num_samples_per_tile
        self.steps_per_epoch = steps_per_epoch
        self.data_loc = data_loc
        self.lead_frames_x = lead_frames_x
        self.lead_frames_y = lead_frames_y
        self.dataType = dataType
        self.tileSize = tileSize

        #sest counters to 0
        #batches_in_tile_counter tracks how many batches have already iterated through from a single tile
        self.batches_in_tile_counter = 0
        #tile_counter tracks how many tiles have already been iterated through
        self.tile_counter = 0


    #################################################################################
    # function: __len__                                                             #
    # description: returns number of steps per epoch, based on calculation outside  #
    #              of function                                                      #
    # input:                                                                        #
    # 1) self: class instance                                                       #
    ################################################################################# 
    def __len__(self):
        return self.steps_per_epoch - 1

    #################################################################################
    # function: __getitem__                                                         #
    # description: returns one batch worth of data. If the dataLoader reaches the   #
    #              end of a file, then the batch is a smaller size than batch_size  #
    # input:                                                                        #
    # 1) self: class instance                                                       #
    ################################################################################# 
    def __getitem__(self):

        #check if the data stored in the object is empty (is empty after object initialization)
        #if self.data is empty, then a file is loaded using numpy's mmap mode (so it is not entirely loaded into memory)
        if (self.data.shape == (1,)):
            self.data = np.load(self.data_loc + self.dataType + '-t-' + str(self.all_tiles[self.tile_counter]) + '.npy', mmap_mode='r')

        self.batches_in_tile_counter += 1

        #create batch (if batch size does not fit in dataset. i.e, your at the end of the file)
        # in this case, the batch will have a size smaller than batch_size
        if (self.batches_in_tile_counter) * self.batch_size >= self.num_samples_per_tile[self.tile_counter]:
            x = self.data[(self.batches_in_tile_counter-1) * self.batch_size:-1, 0:self.lead_frames_x]
            y = self.data[(self.batches_in_tile_counter-1) * self.batch_size:-1, self.lead_frames_x:self.lead_frames_x + self.lead_frames_y]

            #remove data in self.data and update counters accordingly
            self.data = np.array([-1])
            self.batches_in_tile_counter = 0
            self.tile_counter += 1

            #return batch, in form of x and y data
            return x, y

        #create batch (if batch size fits in dataset)
        # in this case, batch will have size equal to batch_size
        else:
            x = self.data[(self.batches_in_tile_counter-1) * self.batch_size:self.batches_in_tile_counter * self.batch_size, 0:self.lead_frames_x]
            y = self.data[(self.batches_in_tile_counter-1) * self.batch_size:self.batches_in_tile_counter * self.batch_size, self.lead_frames_x: self.lead_frames_x + self.lead_frames_y]
            
            #return batch, in form of x and y data
            return x, y

    #################################################################################
    # function: __getitem__                                                         #
    # description: resets counters and removes data in self.data at end of epoch    #
    # input:                                                                        #
    # 1) self: class instance                                                       #
    ################################################################################# 
    def on_epoch_end(self):
        self.batches_in_tile_counter = 0
        self.tile_counter = 0
        self.data = np.array([-1])


if __name__ == "__main__":

    #shows how many GPUs are being used
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)


    #batch size for data loaders
    batch_size = 128 

    #number of tiles that are used in the training, validation, and testing data
    num_tiles = 1#64

    #number of epochs
    epochs = 75
    
    #size of each tile
    tileSize = 64

    #location of training, validation, and testing data, determined by main.json
    train_loc = json_params["train_loc"]
    val_loc = json_params["val_loc"]
    test_loc = json_params["test_loc"]

    #lead_time refers to amount of consecutive time (in minutes)
    #lead_time_x and lead_time_y are determined by main.json
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]

    #meta_data_loc is the location of the metadata, determined by main.json
    meta_data_loc = json_params["meta_data_loc"]

    #since the dataset works in intervals of 5 minutes,
    #lead_time_x and lead_time_y need to be divided by 5 for the number of frames
    lead_frames_x = int(lead_time_x/5)
    lead_frames_y = int(lead_time_y/5)


    train_data, val_data, test_data = [], [], []
    
    #read meta data
    meta_data_df = pd.read_csv(meta_data_loc, sep = ',', header = 0)
    
    tileIDs = meta_data_df['tileID']
    train_num_samples_per_tile = meta_data_df['train_num_samples']
    val_num_samples_per_tile = meta_data_df['val_num_samples']
    train_steps_per_epoch = int(np.sum((train_num_samples_per_tile/batch_size).apply(np.ceil)))
    val_steps_per_epoch = int(np.sum((val_num_samples_per_tile/batch_size).apply(np.ceil)))



    #create training and validation data loaders based on customDataLoader class
    train_batch_generator = customDataLoader(tileIDs, train_num_samples_per_tile, batch_size, train_steps_per_epoch, lead_frames_x, lead_frames_y, tileSize, train_loc, 'train')
    val_batch_generator = customDataLoader(tileIDs, val_num_samples_per_tile, batch_size, val_steps_per_epoch, lead_frames_x, lead_frames_y, tileSize, val_loc, 'val')

    #input shape to the convLSTM
    input_shape = (lead_frames_x, tileSize, tileSize, 1)

    #distributed training strategy provided by Tensorflow
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    #min number of layers is 2
    with strategy.scope():
        model = build_model(input_shape, num_layers = 4, filters = 32, kernel_size = 3)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        mse = tf.keras.losses.MeanSquaredError(reduction = keras.losses.Reduction.NONE)

        #################################################################################
        # function: customm_loss                                                        #
        # description: takes the predicted and true y values, and calculates            #
        #              WMSE (weighted MSE function)                                     #
        #              as described in the technical report                             #
        # inputs:                                                                       #
        # 1) y_true: an array of the true labels, or predicted y values.                #
        # 2) y_pred: an array of the predicted labels, or predicted y values.           #
        # 3) tileSize: size of tiles.                                                   #
        #                                                                               #
        #outputs:                                                                       #
        # 1) loss_value: loss value from rmse                                           #
        ################################################################################# 
        def custom_loss(y_true, y_pred, tileSize = 64):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            np_true = y_true.numpy()
            np_true = np.squeeze(np_true)
            np_pred = y_true.numpy()
            np_pred = np.squeeze(np_pred)

            subBatchSize = np_true.shape[0]
            median = int(0.975 * tileSize * tileSize)
            q3 = int(0.93 * tileSize * tileSize)
            outlier = int(0.83 * tileSize * tileSize)
            sample_weights = np.ones(np_true.shape)

            for i in range(subBatchSize):
                npNumZeros = tileSize*tileSize - np.count_nonzero(np_true[i])
                if npNumZeros <= outlier:
                    sample_weights[i] = sample_weights[i] * 285
                elif npNumZeros <= q3:
                    sample_weights[i] = sample_weights[i] * 3
                elif npNumZeros <= median:
                    sample_weights[i] = sample_weights[i] * 2
            per_example_loss = mse(y_true, y_pred)
            average_loss = tf.nn.compute_average_loss(per_example_loss, sample_weight = sample_weights, global_batch_size = GLOBAL_BATCH_SIZE)
            return average_loss



        model.compile(loss=custom_loss, optimizer=opt, run_eagerly=True, metrics = ['mse'])

        early_stopping = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1, 
                        mode='auto'
                    )
        learning_rate = keras.callbacks.LearningRateScheduler(scheduler)

    #print model architecture summary
    print(model.summary())

    history = model.fit(x = train_batch_generator, validation_data = val_batch_generator, epochs=epochs, verbose = 2, callbacks=[early_stopping, learning_rate])

    #save model as .h5 file
    model.save('tmp_model.h5')

    plotTraining(history)
