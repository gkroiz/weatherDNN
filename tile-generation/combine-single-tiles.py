#################################################################################
# file name: combine-tiles.py                                                   #
# author: Gerson Kroiz                                                          #
# file desc: takes the individual tiles as xarrays, and combines them into 12   #
# xarray file (1 for each month)                                                #
#################################################################################

from netCDF4 import Dataset
import time
import xarray as xr
import os
import sys
from json import load as loadf

#################################################################################
# function: monthStringFunc                                                     #
# description: takes month as an integer and creates it as a string.            #
# Ex: 1 -> 01, 10 -> 10                                                         #
# inputs:                                                                       #
# 1) month: integer for the month                                               #
################################################################################# 
def monthStringFunc(month):
    monthStr = ''
    if month > 8:
        monthStr = str(month + 1)
    else:
        monthStr = '0' + str(month + 1)
    return monthStr

#currently, this only works with slurm, as the rank and size are determined by slurm job arguments
#if you are not using slurm, please change rank and size accordingly
#additionally, the year is based on command line argument
if __name__ == "__main__":

    #read rank, size, and year via command line arguments
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])
    year = sys.argv[1]

    #open json file
    with open("tile-generation.json", 'r') as inFile:
        json_params = loadf(inFile)

    #location of where tiles are located
    tiles_dir_loc = json_params['tiles_dir_loc'] + str(year)

    #location of where to save combined tiles
    combined_tiles_dir_loc = json_params['combined_tiles_dir_loc'] + str(year)


    #dimensions of entire dataset and individual tiles
    tile_dim = json_params['tile_dim']
    entire_dataset_dim = json_params['entire_dataset_dim']

    numTilesPerAxis = int(entire_dataset_dim / tile_dim)

    numMonthsInYear = 12

    #make directory for tiles (only on rank 0)
    if rank == 0:
        if not os.path.isdir(combined_tiles_dir_loc):
            os.mkdir(combined_tiles_dir_loc)
            os.chdir(combined_tiles_dir_loc)
            for tile in range(16*16):
                os.mkdir('tile'+str(tile))

    time.sleep(60)

    #iterate through each tile
    for iter in range(int(numTilesPerAxis*numTilesPerAxis/size)):

        #iterate through each month in a year
        for month in range(numMonthsInYear):
            tileID = iter + rank*int(numTilesPerAxis*numTilesPerAxis/size)

            monthStr = monthStringFunc(month)

            files = tiles_dir_loc + '/tile' + str(tileID) + '/t-' + str(tileID) + '-MRMS_PrecipRate_00.00_' + str(year) + monthStr + '*.nc'

            os.chdir(combined_tiles_dir_loc)

            #check that os.system returns something
            exit_val = os.system('cdo -mergetime -cat ' + files + ' ' + combined_tiles_dir_loc + '/tile' + str(tileID) + '/t-' + str(tileID) + '-' + monthStr + '-time-series.nc')

            #error checking
            if exit_val != 0:
                sys.exit('Issue with cdo')


                
#################################################################################
#                                  EOF                                          #
#################################################################################