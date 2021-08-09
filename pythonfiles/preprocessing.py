#################################################################################
# file name: preprocessing.py                                                   #
# author: Gerson Kroiz                                                          #
# file desc: this file does the preprocessing, to prepare the data for main.py  #
# requirements: preprocessing.json                                              #
#################################################################################

import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4
import xarray as xr
import numpy as np
import pickle
import tensorflow.keras as keras
import csv


#################################################################################
# function: genDataset                                                          #
# description: reads the xarray files, which include all time frames,           #
#              and generates samples for the dataset                            #
# inputs:                                                                       #
# 1) years: array of strings, the dataset will consists of samples form the     #
# years in the array                                                            #   
# 2) tile_loc: location of the tiles                                            #
# 3) lead_time_x: amount of consecutive time (in minutes) for x_data (samples)  #
# 4) lead_time_y:amount of consecutive time (in minutes) for y_data (labels)    #
# 5) tileIDs: array of tileIDs that the samples should be based in              #
# outputs:                                                                      #
# 1) numpy_dataset: array of all the samples created                            #
################################################################################# 
def genDataset(years, tile_loc, lead_time_x, lead_time_y, tileIDs = [-1]):
    dataset = []
    

    #for loop for each year in the range of years
    for year in years:
        tiles = []
        
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
            tile_year_loc = tiles_year_loc + 'tile' + str(tiles[tileIndex]) + '/'

            #open anual tile file
            file_loc = tile_year_loc + '/t-' + str(tiles[tileIndex]) + '-' + str(year) + '-time-series.nc'
            fileData = xr.open_dataset(file_loc)


            #go through time series
            counter = 0
            needContinue = False
            continueCounter = 0
            total_time_per_sample = lead_time_x + lead_time_y
            total_num_time_frames = int(total_time_per_sample / 5)
            for index in range(len(fileData.time)):  
                if needContinue == True:
                    continueCounter += 1
                    if continueCounter == total_num_time_frames - 1:
                        continueCounter = 0
                        needContinue = False
                    continue
                    
                time_frame = fileData.time[index]
                #check that you have consecutive tiles

                #check to make sure you don't go out of bounds
                if time_frame.data + np.timedelta64(total_time_per_sample, 'm') >= fileData.time[-1]:
                    break
                
                #check if there are n consecutive time frames, where n = total_num_time_frames
                isValid = False
                if counter + total_num_time_frames >= fileData.PrecipRate_surface.shape[0]:
                    break
                if time_frame.data + np.timedelta64(total_time_per_sample, 'm') == fileData.time[counter+total_num_time_frames]:
                    isValid = True
                if isValid:

                    #check if there are NaN values, if so, do not include the sample
                    if not np.isnan(fileData.PrecipRate_surface[counter:counter+total_num_time_frames]).any():
                        dataset.append(fileData.PrecipRate_surface[counter:counter+total_num_time_frames])
                    counter += total_num_time_frames
                    needContinue = True
                else:
                    counter += 1

                
        numpy_dataset = np.array(dataset)
    return numpy_dataset

#################################################################################
# function: train_val_test_gen                                                  #
# description: uses genDataset to create train, val, and test datasets. Then,   #
#              these datasets are saved as npy files. This is all for 1 tile    #
#              (does not work for several tiles right now)                      #
# inputs:                                                                       #
# 1) tileID: location of the tile                                               #
################################################################################# 
def train_val_test_gen(tileIDs = [-1]):
    with open("preprocessing.json", 'r') as inFile:
        json_params = loadf(inFile)
    #create one variable that includes all of the data

    #years for training data
    train_years = json_params["train_years"]
    val_years = json_params["val_years"]
    test_years = json_params["test_years"]
    lead_time_x = json_params["lead_time_x"]
    lead_time_y = json_params["lead_time_y"]
    meta_data_loc = json_params["meta_data_loc"]

    #directory where tiles are located
    TILESDIR = json_params["tile_dir"]

    #directories where to save the data
    save_train_loc = json_params["train_loc"]
    save_val_loc = json_params["val_loc"]
    save_test_loc = json_params["test_loc"]

    #generate training, validation, and testing data
    train_data, val_data, test_data = [], [], []
    train_data = genDataset(train_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)
    val_data = genDataset(val_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)
    test_data = genDataset(test_years, TILESDIR, lead_time_x, lead_time_y, tileIDs)

    #add dimension to each dataset (needed for convLSTM)
    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    #write meta data information to the csv file
    if not os.path.isfile(meta_data_loc):
        with open(meta_data_loc, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['tileID', 'train_num_samples', 'val_num_samples', 'test_num_samples'])

    else:
        with open(meta_data_loc,'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([str(tileIDs[0]), str(train_data.shape[0]), str(val_data.shape[0]), str(test_data.shape[0])])

    #save files
    np.save(save_train_loc + '/train-t-' + str(tileIDs[0]), train_data)
    np.save(save_val_loc + '/val-t-' + str(tileIDs[0]), val_data)
    np.save(save_test_loc + '/test-t-' + str(tileIDs[0]), test_data)

if __name__ == "__main__":

    with open("preprocessing.json", 'r') as inFile:
        json_params = loadf(inFile)

    num_tiles = json_params["num_tiles"]
    tilesIDs = np.random.randint(0,256, num_tiles)
    for i in range(len(tilesIDs)):
        train_val_test_gen([tilesIDs[i]])
