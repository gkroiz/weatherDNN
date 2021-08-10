#################################################################################
# file name: create-tiles.py                                                    #
# author: Gerson Kroiz                                                          #
# file desc: this file takes the monthly tiles netcdf data and combines them    #
# into 1 file                                                                   #
# requirements: create-tiles.json                                               #
#################################################################################

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

    #read rank, size, and year via command line arguments
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])
    year = sys.argv[1]

    #open json file
    with open("tile-generation.json", 'r') as inFile:
        json_params = loadf(inFile)

    tile_dim = json_params['tile_dim']
    entire_dataset_dim = json_params['entire_dataset_dim']

    numTilesPerAxis = int(entire_dataset_dim / tile_dim)

    combined_tiles_dir_loc = json_params['combined_tiles_dir_loc'] + str(year)

    #iterate through each tile. For each tile, merge monthly files into 1 annual file
    for iter in range(int(numTilesPerAxis*numTilesPerAxis/size)):
            tileID = iter + rank*int(numTilesPerAxis*numTilesPerAxis/size)
            os.chdir(f'{combined_tiles_dir_loc}/tile' + str(tileID))
            
            #merge monthly files into 1 annual file
            exit_val = os.system('cdo -mergetime -cat *.nc t-' + str(tileID) + '-' + str(year) + '-time-series.nc')

            #error checking
            if exit_val != 0:
                sys.exit('Issue with cdo')



#################################################################################
#                                  EOF                                          #
#################################################################################