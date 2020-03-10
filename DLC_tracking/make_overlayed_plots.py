import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from behaviour.tracking.tracking import prepare_tracking_data
data_path = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\shade_arena_exploration\23_01_20_sp9_processed_centre_tracking.h5')
savehpath = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\shade_arena_exploration\23_01_20_sp9_overlayed_hexplot.png')
savetpath = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\shade_arena_exploration\23_01_20_sp9_overlayed_trajectory.png')

bcgd_path = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\shade_arena_exploration\23_01_20_sp8_background.png')
# Load the data after processing
processed_data = pd.read_hdf(data_path)
# get y and y coordinates
x, y = (processed_data['x'], processed_data['y'])

#~~~~ HEXPLOT ~~~~#

# adjust size of the figure
fig, ax = plt.subplots(figsize=(10,10))
# plot hexagon-grid heatmap
plt.hexbin(x,y, bins = 'log', mincnt=1, cmap=plt.cm.jet, alpha=0.4)
print('hexplotted')
# load background
background = plt.imread(bcgd_path)
print('background loaded')
plt.imshow(background, cmap='gray')
# Add ROI
#confined_shade_arena_exploration ROI
#ROI = patches.Rectangle((345,860),550,200,alpha=0.2,linewidth=3,edgecolor='r',facecolor='r')
#shade_arena_exploration ROI
ROI = patches.Rectangle((395,695),430,400,alpha=0.2,linewidth=3,edgecolor='r',facecolor='r')
ax.add_patch(ROI)
print('added ROI')
plt.savefig(savehpath)
print('saved hexplot')
plt.show()

#~~~~ Trajectory ~~~~#

fig, ax = plt.subplots(figsize=(10,10))
# plot hexagon-grid heatmap
ax.plot(x, y, c='y', alpha=0.5, linewidth=0.5)
print('plotted')
# load background
background = plt.imread(bcgd_path)
print('background loaded')
plt.imshow(background, cmap='gray')
#confined_shade_arena_exploration ROI
#ROI = patches.Rectangle((345,860),550,200,alpha=0.2,linewidth=3,edgecolor='r',facecolor='r')
#shade_arena_exploration ROI
ROI = patches.Rectangle((395,695),435,400,alpha=0.2,linewidth=3,edgecolor='r',facecolor='r')
ax.add_patch(ROI)
print('added ROI')
plt.savefig(savetpath)
print('saved trajectory')
plt.show()