import h5py
import cv2
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt 
from behaviour.tracking.tracking import prepare_tracking_data
from coordinate_alignment.Video_Functions import register_arena, get_background

# ~~~~~~~~~~ Prior to running this script, the initial freezing period should be trimmed off using DLC_DATA_TRIMMING.ipynb ~~~~~~~~~~ #

video_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14\video.avi'
bcgd_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14_background.png'
transform_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14_transform.npy'
trimmed_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14\videoDLC_resnet50_large_spidersJan27shuffle1_1030000.h5'

# ~~~~~~~~~~ Align tracking to a common coordinate system ~~~~~~~~~~ #

# Register transform

# Plot & save trajectory before trasnformation 

# Transform DLC tracking

# Plot & save trajectory after trasnformation 

# ~~~~~~~~~~ Process tracking data ~~~~~~~~~~ #

#Plot and save bp likelihood and distance/frame before processing 

#Interpolate over tracking errors

#Plot and save bp likelihood and distance/frame after processing

# ~~~~~~~~~~ Quantifying tracking data ~~~~~~~~~~ #

#Calculate and save the ROI stats


processed_data = prepare_tracking_data(trimmed_path, likelihood_th=0.9, median_filter=False, interpolate_nans=True, common_coord=False, compute=True)
prepare_tracking_data()
print(processed_data['centre'])


