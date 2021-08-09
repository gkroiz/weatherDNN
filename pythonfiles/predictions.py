#################################################################################
# file name: predictions.py                                                     #
# author: Gerson Kroiz                                                          #
# file desc: this model takes the trained model from main.py and conducts       #
# predictions                                                                   #
# requirements: main.py, main.json, preprocessing.py, model.py                  #
#################################################################################

import json
from matplotlib.colors import Normalize
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
import matplotlib
from matplotlib import pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from csv import reader
from tensorflow.keras.layers import LeakyReLU
from matplotlib import ticker

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

if __name__ == "__main__":

    with open("predictions.json", 'r') as inFile:
        json_params = loadf(inFile)

    model_loc = json_params['model_loc']
    #DATA LOCATION
    DATE = '08-16-2005'
    DATALOC = '/raid/gkroiz1/gtDays/t-194-64-' + DATE + '.npy'
    SAVEPREDLOC = '/raid/gkroiz1/predDaysComplex/t-194-64-' + DATE + '-pred.npy'
    SAVEGTLOC = '/raid/gkroiz1/gtDaysComplex/t-194-64-' + DATE + '-gt.npy'
    #mse
    #custom

    NETCDFLOC = '/raid/gkroiz1/netcdfFiles/'
    PLOTSLOC = '/home/gkroiz1/weatherDNN/pythonfiles/plots/'

    #load model and custom loss function
    mse = tf.keras.losses.MeanSquaredError(reduction = keras.losses.Reduction.NONE)
    batch_size = 128
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync    
    model = keras.models.load_model(model_loc, custom_objects={'custom_loss': custom_loss})

    lead_time_x = 20
    lead_time_y = 5
    lead_frames_x = int(lead_time_x/5)
    lead_frames_y = int(lead_time_y/5)

    ground_truth = np.load(DATALOC)
    ground_truth = np.expand_dims(ground_truth, axis=-1)

    x_gt = ground_truth[:, 0: lead_frames_x]
    print('x_gt shape: ' + str(x_gt.shape))
    y_gt = ground_truth[:, lead_frames_x: lead_frames_x + lead_frames_y]
    y_gt = np.squeeze(y_gt)
    print('y_gt shape: ' + str(y_gt.shape))


    y_pred = model.predict(x_gt)
    y_pred = np.squeeze(y_pred)



    # np.save(SAVEPREDLOC, y_pred)
    # np.save(SAVEGTLOC, y_gt)

    mmDayData_gt = np.zeros((64,64))
    mmDayData_pred = np.zeros((64,64))

    numNoGT, numLightGT,nnumLightToModGT, numModGT, numModToHeavyGT, numHeavyGT = 0,0,0,0,0,0
    numNoPRED, numLightPRED,nnumLightToModPRED, numModPRED, numModToHeavyPRED, numHeavyPRED = 0,0,0,0,0,0

    print('y_gt.shape[0]: ' + str(y_gt.shape[0]))
    for i in range(y_gt.shape[0]):
        mmDayData_gt += y_gt[i]
        mmDayData_pred += y_pred[i]

        numNoGT += (y_gt[i] < 0.5).sum()
        numLightGT += ((y_gt[i] < 2) & (y_gt[i] >= 0.5)).sum()
        nnumLightToModGT += ((y_gt[i] < 5) & (y_gt[i] >= 2)).sum()
        numModGT += ((y_gt[i] < 10) & (y_gt[i] >= 5)).sum()
        numModToHeavyGT += ((y_gt[i] < 30) & (y_gt[i] >= 10)).sum()
        numHeavyGT += (y_gt[i] >= 30).sum()

        numNoPRED += (y_pred[i] < 0.5).sum()
        numLightPRED += ((y_pred[i] < 2) & (y_pred[i] >= 0.5)).sum()
        nnumLightToModPRED += ((y_pred[i] < 5) & (y_pred[i] >= 2)).sum()
        numModPRED += ((y_pred[i] < 10) & (y_pred[i] >= 5)).sum()
        numModToHeavyPRED += ((y_pred[i] < 30) & (y_pred[i] >= 10)).sum()
        numHeavyPRED += (y_pred[i] >= 30).sum()

    mmDayData_gt = mmDayData_gt * 24 / (y_gt.shape[0])
    mmDayData_pred = mmDayData_pred * 24 / (y_gt.shape[0])

    print('total = ' + str(numNoGT + numLightGT + nnumLightToModGT + numModGT + numModToHeavyGT + numHeavyGT))
    print('total = ' + str(numNoPRED + numLightPRED + nnumLightToModPRED + numModPRED + numModToHeavyPRED + numHeavyPRED))

    numNoGT = numNoGT / (y_gt.shape[0] * 64 * 64) * 100
    numLightGT = numLightGT / (y_gt.shape[0] * 64 * 64) * 100
    nnumLightToModGT = nnumLightToModGT / (y_gt.shape[0] * 64 * 64) * 100
    numModGT = numModGT / (y_gt.shape[0] * 64 * 64) * 100
    numModToHeavyGT = numModToHeavyGT / (y_gt.shape[0] * 64 * 64) * 100
    numHeavyGT = numHeavyGT / (y_gt.shape[0] * 64 * 64) * 100

    numNoPRED = numNoPRED / (y_gt.shape[0] * 64 * 64) * 100
    numLightPRED = numLightPRED / (y_gt.shape[0] * 64 * 64) * 100
    nnumLightToModPRED = nnumLightToModPRED / (y_gt.shape[0] * 64 * 64) * 100
    numModPRED = numModPRED / (y_gt.shape[0] * 64 * 64) * 100
    numModToHeavyPRED = numModToHeavyPRED / (y_gt.shape[0] * 64 * 64) * 100 
    numHeavyPRED = numHeavyPRED / (y_gt.shape[0] * 64 * 64) * 100

    print('numNoGT: ' + str(numNoGT))
    print('numLightGT: ' + str(numLightGT))
    print('nnumLightToModGT: ' + str(nnumLightToModGT))
    print('numModGT: ' + str(numModGT))
    print('numModToHeavyGT: ' + str(numModToHeavyGT))
    print('numHeavyGT: ' + str(numHeavyGT))

    print('numNoPRED: ' + str(numNoPRED))
    print('numLightPRED: ' + str(numLightPRED))
    print('nnumLightToModPRED: ' + str(nnumLightToModPRED))
    print('numModPRED: ' + str(numModPRED))
    print('numModToHeavyPRED: ' + str(numModToHeavyPRED))
    print('numHeavyPRED: ' + str(numHeavyPRED))

    # mmDayData_gt = np.rot90(np.fliplr(mmDayData_gt))
    # mmDayData_pred = np.rot90(np.fliplr(mmDayData_pred))
    print('y_pred_avg: ' + str(np.mean(mmDayData_pred)/24))
    print('y_gt_avg: ' + str(np.mean(mmDayData_gt)/24))
    print('y_pred_max: ' + str(np.max(mmDayData_pred)/24))
    print('y_gt_max: ' + str(np.max(mmDayData_gt)/24))
    plt.figure(1)
    plt.imshow(mmDayData_gt, cmap='coolwarm')
    # gtplot.set_clim(0.,.03)
    plt.axis('off')
    cbar = plt.colorbar()
    tick_font_size = 16
    cbar.ax.tick_params(labelsize=tick_font_size)    
    tick_locator = ticker.MaxNLocator(nbins=8, min_n_ticks=8)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # plt.title('Observed Light (08/16/2005)', fontsize=22, pad=15, loc='left')
    # plt.title('Observed Medium (08/13/2005)', fontsize=22, pad=15, loc='left')
    plt.title('Observed Heavy (08/15/2005)', fontsize=22, pad=15, loc='left')
    # plt.title(dtstr, loc='right')
    plt.tight_layout()
    # plt.savefig(PLOTSLOC + 't-194-64-' + DATE + '-gt.pdf')


    plt.figure(2)
    plt.imshow(mmDayData_pred, cmap='coolwarm')
    plt.axis('off')
    # gtplot.set_clim(0.,.03)
    cbar = plt.colorbar()
    tick_font_size = 16
    cbar.ax.tick_params(labelsize=tick_font_size)    
    tick_locator = ticker.MaxNLocator(nbins=8, min_n_ticks=8)
    cbar.locator = tick_locator
    cbar.update_ticks()
# LinearLocator(5)
    # plt.title('Predicted Light (08/16/2005)', fontsize=22, pad= 15, loc='left')
    # plt.title('Predicted Medium (08/13/2005)', fontsize=22, pad= 15, loc='left')
    plt.title('Predicted Heavy (08/15/2005)', fontsize=22, pad= 15, loc='left')
    # plt.title(dtstr, loc='right')
    plt.tight_layout()
    # plt.savefig(PLOTSLOC + 't-194-64-' + DATE + '-pred.pdf')

    # dataArray_gt = xr.DataArray(data=mmDayData_gt)
    # xrmmDayData_gt = xr.Dataset(data_vars=dict(PrecipRate_surface=(["longitude","latitude"], dataArray_gt))
    # # coords = dict(latitude=data.latitude,longitude=data.longitude,time=data.time),
    # )

    # dataArray_pred = xr.DataArray(data=mmDayData_pred)
    # xrmmDayData_pred = xr.Dataset(data_vars=dict(PrecipRate_surface=(["longitude","latitude"], dataArray_pred))
    # # coords = dict(latitude=data.latitude,longitude=data.longitude,time=data.time),
    # )

    # xrmmDayData_gt.to_netcdf(NETCDFLOC  + '/t-194-64-' + DATE + '-gt.nc', mode='w', format='netcdf4')
    # xrmmDayData_pred.to_netcdf(NETCDFLOC  + '/t-194-64-' + DATE + '-pred.nc', mode='w', format='netcdf4')



