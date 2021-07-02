import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4
import xarray as xr
import numpy as np

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
            print('tile: ' + str(index))
            tile_year_loc = tiles_year_loc + 'tile' + str(index) + '/'
            # sample = []

            #open anual tile file
            file_loc = tile_year_loc + '/t-' + str(tiles[index]) + '-' + str(year) + '-time-series.nc'
            fileData = xr.open_dataset(f'{file_loc}')

            #go through time series
            # for time_frame in range(fileData.PrecipRate_surface.shape[0]):
            counter = 0
            for time_frame in fileData.time:  

                #check that you have consecutive tiles
                total_time_per_sample = lead_time_sample + lead_time_label
                total_num_time_frames = int(total_time_per_sample / 5)
                

                #check to make sure you don't go out of bounds
                if time_frame + np.timedelta64(total_num_time_frames, 'm') >= fileData.time[-1]:
                    break
                
                #check if there are n consecutive time frames, where n = total_num_time_frames
                isValid = False
                if time_frame + np.timedelta64(total_time_per_sample, 'm') == fileData.time[counter]:
                    isValid = True

                if isValid:
                    np.append(dataset, fileData.PrecipRate_surface[time_frame:time_frame+total_num_time_frames])
                    counter += total_num_time_frames
                    time_frame += np.timedelta64(total_num_time_frames, 'm')
                else:
                    counter += 1

                

    return dataset

# class InputData:
#     def __init__(self, data, lead_time_sample, lead_time_label):
#         #(lead_time * 12, 64, 64)
#         self.data = data

#         self.lead_time_sample = lead_time_sample
#         self.lead_time_label = lead_time_label



if __name__ == "__main__":


    #create one variable that includes all of the data

    #years for training data
    train_years = ['2001']

    #directory where tiles are located
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/combined-tiles/'
    #call data function

    start = time.time()

    train_data = genData(train_years, TILESDIR, 20, 5, [0])
    end = time.time()

    np.save('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/pythonfiles/train_data.npy', train_data)

    print('after train data')

    print(train_data.shape)

    print('time elapsed = ' + str(end - start))

    # model = build_model
    # model.compile()

    # model.fit()
