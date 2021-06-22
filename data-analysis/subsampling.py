import numpy as np
import xarray as xr
# import 


if __name__ == "__main__":
    DATADIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/2004'
    data = xr.open_mfdataset(f'{DATADIR}/*.nc', combine='by_coords')

    print(data)