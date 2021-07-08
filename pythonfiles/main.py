import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4
import xarray as xr
import numpy as np
import pickle
from model import build_model
from matplotlib import pyplot as plt
import tensorflow.keras as keras


# from model import build_model

#tf.records, faster than other formats for I/O data


#label should be 1 64x64
#lead_time_sample and lead_time_label should be in terms of minutes
def genData(years, tile_loc, lead_time_sample, lead_time_label, tileIDs = [-1]):
    dataset = []
    tiles = []

    #for loop for each year in the range of years
    for year in years:
        #check which tiles to add to dataset
        if tileIDs[0] == -1:
            tiles = list(range(0, 256))
        else:
            for index in tileIDs:
                tiles.append(index)

        #location of the tiles
        tiles_year_loc = tile_loc + '/' + year + '/'

        #index through each of the selected tiles
        for index in tiles:
            # print('tile: ' + str(index))
            tile_year_loc = tiles_year_loc + 'tile' + str(index) + '/'
            # sample = []

            #open anual tile file
            file_loc = tile_year_loc + '/t-' + str(tiles[index]) + '-' + str(year) + '-time-series.nc'
            fileData = xr.open_dataset(file_loc)
            # print('final time: ' + str(fileData.time[-1]))

            #go through time series
            # for time_frame in range(fileData.PrecipRate_surface.shape[0]):
            counter = 0
            # continue_counter = 
            needContinue = False
            continueCounter = 0
            total_time_per_sample = lead_time_sample + lead_time_label
            total_num_time_frames = int(total_time_per_sample / 5)
            for index in range(len(fileData.time)):  
                if needContinue == True:
                    continueCounter += 1
                    if continueCounter == total_num_time_frames - 1:
                        continueCounter = 0
                        needContinue = False
                    continue
                    
                # print('index: ' + str(index))
                time_frame = fileData.time[index]
                #check that you have consecutive tiles

                #check to make sure you don't go out of bounds
                # print('ctouner: ' + str(counter))
                # print('time: ' + str(time_frame + np.timedelta64(total_time_per_sample, 'm')))
                # print('counter time: ' + str(fileData.time[counter+total_num_time_frames]))

                if time_frame.data + np.timedelta64(total_time_per_sample, 'm') >= fileData.time[-1]:
                    break
                
                #check if there are n consecutive time frames, where n = total_num_time_frames
                isValid = False
                if counter + total_num_time_frames >= fileData.PrecipRate_surface.shape[0]:
                    break
                if time_frame.data + np.timedelta64(total_time_per_sample, 'm') == fileData.time[counter+total_num_time_frames]:
                    isValid = True
                # print('isValid: ' + str(isValid))
                if isValid:

                    #check if there are NaN values, if so, do not include
                    if not np.isnan(fileData.PrecipRate_surface[counter:counter+total_num_time_frames]).any():
                        dataset.append(fileData.PrecipRate_surface[counter:counter+total_num_time_frames])
                    counter += total_num_time_frames
                    # time_frame.data += np.timedelta64(total_time_per_sample, 'm')
                    # index += total_num_time_frames
                    needContinue = True
                    # print('index in isValid:' + str(index))
                    # print('time in isValid: ' + str(time_frame + np.timedelta64(total_time_per_sample, 'm')))
                else:
                    counter += 1

                
        numpy_dataset = np.array(dataset)
    return numpy_dataset

# class InputData:
#     def __init__(self, data, lead_time_sample, lead_time_label):
#         #(lead_time * 12, 64, 64)
#         self.data = data

#         self.lead_time_sample = lead_time_sample
#         self.lead_time_label = lead_time_label

def plotTraining(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('trainingplot.pdf')

if __name__ == "__main__":

    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)
    #create one variable that includes all of the data

    #years for training data
    train_years = json_params["train_years"]
    val_years = json_params["val_years"]
    test_years = json_params["test_years"]
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]

    lead_frames_x = int(lead_time_x/5)
    lead_frames_y = int(lead_time_y/5)

    #directory where tiles are located
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/combined-tiles/'
    #call data function

    # lead_time_x = 20
    # lead_time_y = 5

    start = time.time()

    train_loc = '/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/train_data.npy'
    val_loc = '/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/val_data.npy'
    test_loc = '/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/test_data.npy'


    train_data, val_data, test_data = [], [], []
    if os.path.isfile(train_loc):
        train_data = np.load(train_loc)
    else:
        train_data = genData(train_years, TILESDIR, lead_time_x, lead_time_y, [0])
        np.save('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/train_data.npy', train_data)
    if os.path.isfile(val_loc):
       val_data = np.load(val_loc)
    else:
        val_data = genData(val_years, TILESDIR, lead_time_x, lead_time_y, [0])
        np.save('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/val_data.npy', val_data)
    if os.path.isfile(test_loc):
        test_data = np.load(test_loc)
    else:
        test_data = genData(test_years, TILESDIR, lead_time_x, lead_time_y, [0])
        np.save('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/test_data.npy', test_data)
    end = time.time()

    # xr.Dataset.to_netcdf('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/train_data.npy', train_data)
    # print('after train data')
    
    print('data gen time = ' + str(end - start))

    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    #normalize data:
    max_val = max(np.max(train_data), np.max(val_data))

    train_data = train_data/max_val
    val_data = val_data/max_val
    test_data = test_data/max_val

    train_x = train_data[:,0:lead_frames_x]
    train_y = train_data[:,lead_frames_x-1:lead_frames_x+lead_frames_y-1]

    val_x = val_data[:,0:lead_frames_x]
    val_y = val_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]

    test_x = test_data[:,0:lead_frames_x]
    test_y = test_data[:,lead_frames_x-1:lead_frames_x + lead_frames_y-1]

    print('before reshaping')
    # train_x = train_x.reshape((train_x.shape[0], lead_time_x, 64, 64))
    # train_y = train_y.reshape((train_y.shape[0], lead_time_y))

    model = build_model(train_x, num_layers = 1, filters = 64, kernel = 3)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=opt, metrics =['mse', 'accuracy'])

    print(model.summary())
    history = model.fit(train_x, train_y, batch_size=512, validation_data=[val_x, val_y], epochs=50,
        callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=2,
                        verbose=1, 
                        mode='auto'
                    )])

    model.save('trained_model.h5')

    #load model
    # keras.models.load_model('trained_model.h5')


    plotTraining(history)