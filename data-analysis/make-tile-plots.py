import numpy as np
import xarray as xr
import time
import os, os.path
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl


if __name__ == "__main__":

    year = sys.argv[1]
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/tile-sums/' + str(year)

    plt.figure(0)
    start = time.time()
    mpl.rcParams['axes.linewidth'] = 0.1
    for tileID in range(256):
        sums = np.load(f'{TILESDIR}/tile'+str(tileID)+'/sums.npy')
        num_files = np.load(f'{TILESDIR}/tile'+str(tileID)+'/num_files.npy')
        SMALLSIZE = 1
        INBETWEENSIZE = 0.3
        TINYSIZE = 0.1
        FONTSIZE = 0.05
        ax = plt.subplot2grid((16,16), (int(tileID/16),tileID%16))
        plt.rc('font', size=INBETWEENSIZE) 
        plt.rc('figure', titlesize=SMALLSIZE)
        # plt.rc('xtick', labelsize=SMALLSIZE)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize=SMALLSIZE)
        plt.plot(sums/(64*64*num_files[0]), linewidth = 0.2)
        plt.ylim([10**(-10), 10**0])
        plt.xlim([0,160])
        plt.yscale('log')
        # plt.ylabel('PDF')
        # plt.xlabel('Precipitation Rate (mm/hr)')
        plt.title('tile' + str(tileID), pad=-50)
        plt.locator_params(axis='y', numticks=10)
        plt.locator_params(axis='x', nbins=8)
        plt.xticks(fontsize = TINYSIZE)
        plt.yticks(fontsize = TINYSIZE)
        ax.tick_params(width = TINYSIZE, length = INBETWEENSIZE, pad = -0.1, labelsize = 0.01)
        # plt.locator_params(axis='y', numticks=20)


    os.chdir('/gpfs/alpine/cli900/world-shared/users/gkroiz1/tile-plots/' + str(year))
    plt.savefig('line-hist-tiles-' + str(year) + '.pdf')

    end = time.time()

    print('time elapsed = ' + str(end - start))
