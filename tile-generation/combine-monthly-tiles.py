#################################################################################
# file name: create-tiles.py                                                    #
# author: Gerson Kroiz                                                          #
# file desc: this file takes the monthly tiles netcdf data and combines them    #
# into 1 file                                                                   #
# requirements: create-tiles.json                                               #
#################################################################################

# Description: takes individual tiles and combines them into larger netcdf files by month for each year

import netCDF4
from netCDF4 import Dataset
import time
import xarray as xr
import os
import sys
from json import load as loadf

#currently, this only works with slurm scripts, as the rank and size are determined by slurm job arguments
#if you are not using slurm, please change rank and size accordingly
#additionally, the year is based on command line argument
if __name__ == "__main__":

    year = sys.argv[1]
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])

    tileDim = 256
    numTilesPerAxis = int(1024 / tileDim)

    with open("tile-generation.json", 'r') as inFile:
        json_params = loadf(inFile)  

    combined_tiles_dir_loc = json_params['combined_tiles_dir_loc'] + str(year)


    start = time.time()
    for iter in range(int(numTilesPerAxis*numTilesPerAxis/size)):
            tileID = iter + rank*int(numTilesPerAxis*numTilesPerAxis/size)
            os.chdir(f'{combined_tiles_dir_loc}/tile' + str(tileID))

            os.system('cdo -mergetime -cat *.nc t-' + str(tileID) + '-' + str(year) + '-time-series.nc')
    end = time.time()
    
    print('rank: ' + str(rank) + ', time elapsed = ' + str(end - start))

