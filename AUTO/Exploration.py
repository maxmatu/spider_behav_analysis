import h5py
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from behaviour.tracking.tracking import prepare_tracking_data
fro
trimmed_data = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14\videoDLC_resnet50_large_spidersJan27shuffle1_1030000.h5'



processed_data = prepare_tracking_data(trimmed_data, likelihood_th=0.9, median_filter=False, interpolate_nans=True, common_coord=False, compute=True)
prepare_tracking_data()
print(processed_data['centre'])


