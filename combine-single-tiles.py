#################################################################################
# file name: combine-tiles.py                                                   #
# author: Gerson Kroiz                                                          #
# file desc: takes the tiles as xarrays, and combines them into 12 xarray file  #
# (1 for each month)                                                            #
#################################################################################

# Description: takes individual tiles and combines them into larger netcdf files by month for each year

from netCDF4 import Dataset
import time
import xarray as xr
import os
import sys
from json import load as loadf

if __name__ == "__main__":

    year = sys.argv[1]
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])

    with open("tile-generation.json", 'r') as inFile:
        json_params = loadf(inFile)

    tiles_dir_loc = json_params['tiles_dir_loc'] + str(year)


   
    combined_tiles_dir_loc = json_params['combined_tiles_dir_loc'] + str(year)

    if rank == 0:
        if not os.path.isdir(combined_tiles_dir_loc):
            os.mkdir(combined_tiles_dir_loc)
            os.chdir(combined_tiles_dir_loc)
            for tile in range(16*16):
                os.mkdir('tile'+str(tile))

    time.sleep(60)


    start = time.time()
    for iter in range(int(256/size)):
        for month in range(12):
            tileID = iter + rank*int(256/size)
            monthStr = ''
            if month > 8:
                monthStr = str(month + 1)
            else:
                monthStr = '0' + str(month + 1)

            files = tiles_dir_loc + '/tile' + str(tileID) + '/t-' + str(tileID) + '-MRMS_PrecipRate_00.00_' + str(year) + monthStr + '*.nc'
            os.chdir(combined_tiles_dir_loc)

            os.system('cdo -mergetime -cat ' + files + ' ' + combined_tiles_dir_loc + '/tile' + str(tileID) + '/t-' + str(tileID) + '-' + monthStr + '-time-series.nc')
    end = time.time()
    
    print('rank: ' + str(rank) + ', time elapsed = ' + str(end - start))

