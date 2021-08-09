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


#currently, this only works with slurm scripts, as the rank and size are determined by slurm job arguments
#however, this can easily be adjusted by setting rank and size accordingly (not based on slurm)
#additionally, the year is based on command line argument
if __name__ == "__main__":
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])
    

    year = sys.argv[1]

    with open("tile-generation.json", 'r') as inFile:
        json_params = loadf(inFile)
    
    #location of xarray 1024x1024 data
    data_dir_loc = json_params['data_dir_loc']

    #location of where to save tiles
    tiles_dir_loc = json_params['tiles_dir_loc']

    tile_dim = json_params['tile_dim']
    numTilesPerAxis = int(1024 / tile_dim)
    

    #find number of files in data directory (should be 5 * 12 * 24)
    num_files = len([name for name in os.listdir(data_dir_loc) if os.path.isfile(os.path.join(data_dir_loc, name))])


    #make directory for tiles
    if rank == 0:
        # print("Here, rank: " + str(rank))
        if not os.path.isdir(tiles_dir_loc):
            os.mkdir(tiles_dir_loc)
            os.chdir(tiles_dir_loc)
            for tile in range(numTilesPerAxis*numTilesPerAxis):
                os.mkdir('tile'+str(tile))
    
    time.sleep(60)

    overallStart = time.time()

    #goes through each file
    for filename in os.listdir(data_dir_loc):
        # start = time.time()

        data = xr.open_dataset(f'{data_dir_loc}/{filename}')

        #goes through each latitude coord
        for lat in range(int(numTilesPerAxis/size)):

            #goes through each longitude coord
            for long in range(numTilesPerAxis):
                mpi_lat = lat + int(rank)*int(numTilesPerAxis/size)

                #make a subdataset of the lat, long, and prec of the tile
                tile = xr.Dataset(
                    data_vars=dict(
                        PrecipRate_surface=data.PrecipRate_surface[0, 1024-mpi_lat*tile_dim-tile_dim:1024-mpi_lat*tile_dim, long*tile_dim:long*tile_dim+tile_dim]
                    ),
                    coords = dict(
                       time=data.time
                    ),
                    attrs = dict(Conventions='COARDS', History='created by wgrib2', GRIB2_grid_template = '0')
                    )

                #save dataset
                os.chdir(tiles_dir_loc + '/tile' + str(mpi_lat * numTilesPerAxis + long))
                tile.to_netcdf(f't-' + str(mpi_lat * numTilesPerAxis + long) + '-' + str(filename), mode='w', format='netcdf4')


    overallEnd = time.time()
    print('year: ' + str(year) + ', rank: ' + str(rank) + ', FINAL TIME = ' + str(overallEnd - overallStart))

