#################################################################################
# file name: create-tiles.py                                                    #
# author: Gerson Kroiz                                                          #
# file desc: this file takes the location of the 1024x1024 xarray files,        #
# and creates tiles as xarray files                                             #
# requirements: create-tiles.json                                               #
#################################################################################

import numpy as np
import xarray as xr
import time
import os, os.path
import sys
from json import load as loadf


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
    
    #location of xarray 1024x1024 data
    data_dir_loc = json_params['data_dir_loc']

    #location of where to save tiles
    tiles_dir_loc = json_params['tiles_dir_loc'] + '/' + str(year)

    #dimensions of entire dataset and individual tiles
    tile_dim = json_params['tile_dim']
    entire_dataset_dim = json_params['entire_dataset_dim']

    numTilesPerAxis = int(1024 / tile_dim)
    

    #find number of files in data directory (should be 5 * 12 * 24)
    num_files = len([name for name in os.listdir(data_dir_loc) if os.path.isfile(os.path.join(data_dir_loc, name))])


    #make directory for tiles (only on rank 0)
    if rank == 0:
        if not os.path.isdir(tiles_dir_loc):
            os.mkdir(tiles_dir_loc)
            os.chdir(tiles_dir_loc)
            for tile in range(numTilesPerAxis*numTilesPerAxis):
                os.mkdir('tile'+str(tile))
    
    time.sleep(60)

    #go through each netcdf file for a year. For each netcdf file, divide this into tiles
    for filename in os.listdir(data_dir_loc):

        data = xr.open_dataset(f'{data_dir_loc}/{filename}')

        #goes through each latitude coord based on tile size
        for lat in range(int(numTilesPerAxis/size)):

            #goes through each longitude coord based on tile size
            for long in range(numTilesPerAxis):
                tile_lat = lat + int(rank)*int(numTilesPerAxis/size)

                #make a tile of the lat, long, and prec of the tile
                #attributes are to keep consistency with the entire dataset netcdf files
                tile = xr.Dataset(
                    data_vars=dict(
                        PrecipRate_surface=data.PrecipRate_surface[0, entire_dataset_dim-tile_lat*tile_dim-tile_dim:entire_dataset_dim-tile_lat*tile_dim, long*tile_dim:long*tile_dim+tile_dim]
                    ),
                    coords = dict(
                       time=data.time
                    ),
                    attrs = dict(Conventions='COARDS', History='created by wgrib2', GRIB2_grid_template = '0')
                    )

                #save tile
                os.chdir(tiles_dir_loc + '/tile' + str(tile_lat * numTilesPerAxis + long))
                tile.to_netcdf(f't-' + str(tile_lat * numTilesPerAxis + long) + '-' + str(filename), mode='w', format='netcdf4')



#################################################################################
#                                  EOF                                          #
#################################################################################
