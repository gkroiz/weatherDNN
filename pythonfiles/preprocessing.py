import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4
import xarray as xr
import numpy as np
import pickle
from matplotlib import pyplot as plt
import tensorflow.keras as keras



#tf.records, faster than other formats for I/O data

#label should be 1 64x64
#lead_time_sample and lead_time_label should be in terms of minutes
def genDataset(years, tile_loc, lead_time_sample, lead_time_label, tileIDs = [-1]):
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
        for tileIndex in range(len(tiles)):
            # print('tile: ' + str(index))
            tile_year_loc = tiles_year_loc + 'tile' + str(tiles[tileIndex]) + '/'
            # sample = []

            #open anual tile file
            file_loc = tile_year_loc + '/t-' + str(tiles[tileIndex]) + '-' + str(year) + '-time-series.nc'
            fileData = xr.open_dataset(file_loc)

            print('size of .nc file: ' + str(fileData.PrecipRate_surface.shape))
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


#uses genDataset to create train, val, test, and save it to npy
def train_val_test_gen(train_loc, val_loc, test_loc, tileIDs = [-1]):
    with open("preprocessing.json", 'r') as inFile:
        json_params = loadf(inFile)
    #create one variable that includes all of the data

    #years for training data
    train_years = json_params["train_years"]
    val_years = json_params["val_years"]
    test_years = json_params["test_years"]
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]

    #directory where tiles are located
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/combined-tiles/'
    #call data function

    # lead_time_x = 20
    # lead_time_y = 5

    start = time.time()

    train_loc = json_params["train_loc"]
    val_loc = json_params["val_loc"]
    test_loc = json_params["test_loc"]

    train_data, val_data, test_data = [], [], []
    train_data = genDataset(train_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)
    val_data = genDataset(val_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)
    test_data = genDataset(test_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)

    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    #randomly take only 10% of the data
    train_idx = np.random.randint(train_data.shape[0], size = int(train_data.shape[0]/10))
    train_data = train_data[train_idx, :]

    val_idx = np.random.randint(val_data.shape[0], size = int(val_data.shape[0]/10))
    val_data = val_data[val_idx, :]

    test_idx = np.random.randint(test_data.shape[0], size = int(test_data.shape[0]/10))
    test_data = test_data[test_idx, :]

    print('new shapes')
    print('train.shape()' + str(train_data.shape))
    print('val.shape()' + str(val_data.shape))
    print('test.shape()' + str(test_data.shape))

    #normalize data:
    # max_val = max(np.max(train_data), np.max(val_data))
    # train_data = train_data/max_val
    # val_data = val_data/max_val
    # test_data = test_data/max_val

    np.save(train_loc, train_data)
    np.save(val_loc, val_data)
    np.save(test_loc, test_data)

    end = time.time()

    print('data gen time = ' + str(end - start))


