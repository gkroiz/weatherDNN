import pandas as pd
import os
import time
from json import load as loadf
import netCDF4 as nc4

from model import build_model

if __name__ == "__main__":
    model = build_model
    model.compile()

    model.fit()
