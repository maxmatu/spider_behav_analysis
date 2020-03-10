import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from roi_analysis import get_timeinrois_stats
import matplotlib.patches as patches

# Load tracking
corrected_posedata = pd.read_hdf('C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_exploration\\02_02_20_sp2_processed_centre_tracking.h5')

# Get XY tracking
x = corrected_posedata.x.values
y = corrected_posedata.y.values
xy = np.vstack([x, y]).T # creates an Nx2 array with N=number of frames and XY at each frame

# Define ROI position
position = namedtuple('position', ['topleft', 'bottomright'])
rois = {'shelter': position((375,825), (850,1025)),}

# Get ROI stats
res = get_timeinrois_stats(xy , rois, fps=40,  returndf=False, check_inroi=False)
print(res)