# Description: takes individual tiles and combines them into larger netcdf files by month for each year

import netCDF4
from netCDF4 import Dataset
import time
import xarray as xr
import os
import sys

if __name__ == "__main__":

    year = sys.argv[1]
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])

    #Full tiles dir
#     TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/tiles/' + str(year)


    #Test tiles dir:
    # TILESDIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/tiles'
   
    COMBINEDTILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/combined-tiles/' + str(year)

    # tile = 1

    start = time.time()
    for iter in range(int(256/size)):
            tileID = iter + rank*int(256/size)
            # files = TILESDIR + '/tile' + str(tileID) + '/t-' + str(tileID) + '-MRMS_PrecipRate_00.00_' + str(year) + monthStr + '*.nc'
            # print("files: " + files)
            os.chdir(f'{COMBINEDTILESDIR}/tile' + str(tileID))
            # os.system('ncrcat ' + files + ' ' + COMBINEDTILESDIR + '/tile' + str(tileID) + '/t-' + str(tileID) + '-time-series.nc')

            os.system('cdo -mergetime -cat *.nc t-' + str(tileID) + '-' + str(year) + '-time-series.nc')
    end = time.time()
    
    # print('year: ' + str(year) + ', rank: ' + str(rank) + ', FINAL TIME = ' + str(overallEnd - overallStart))
    print('rank: ' + str(rank) + ', time elapsed = ' + str(end - start))

