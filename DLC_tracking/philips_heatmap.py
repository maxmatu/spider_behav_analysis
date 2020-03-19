import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from behaviour.tracking.tracking import prepare_tracking_data
import cv2
# Load background image
data_path = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_exploration\06_02_20_sp13_processed_centre_tracking.h5')
bcgd_path = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_exploration\06_02_20_sp13_background.png')
save_path = (r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_exploration\06_02_20_sp13_overlayed_PS_heatmap.png')
# Load the data after processing
processed_data = pd.read_hdf(data_path)
# get y and y coordinates
x, y = (processed_data['x'], processed_data['y'])
print('data loaded')
# Enter parameters
saturation_percentile = 100git
color_dimming = .67
dots_multiplier = .8
# Enter coordinates
position = (x, y)
# Enter height and width of image
height, width = 1152,1216
# Histogram of positions
H, x_bins, y_bins = \
    np.histogram2d(position[0], position[1], [np.arange(0, width + 1), np.arange(0, height + 1)], normed=True)
# Transpose H (so it looks like a proper image)
H = H.T
# make an image out of it
exploration_image = H.copy()
# get it into a reasonable value scale
exploration_image = (exploration_image / np.percentile(exploration_image, 99) * 255)
exploration_image[exploration_image > 255] = 255
# gaussian blur the image
exploration_blur = cv2.GaussianBlur(exploration_image, ksize=(201, 201), sigmaX=15, sigmaY=15)
# normalize and saturate
exploration_blur = (exploration_blur /  np.percentile(exploration_blur, saturation_percentile) * 255)
exploration_blur[exploration_blur > 255] = 255
# change color map
exploration_blur = exploration_blur * color_dimming
exploration_blur = cv2.applyColorMap(255 - exploration_blur.astype(np.uint8), cv2.COLORMAP_HOT)
# make composite image
exploration_blur[H > 0] = exploration_blur[H > 0] * [dots_multiplier, dots_multiplier, dots_multiplier]
# show image
print('plotting')
cv2.imshow('heat map', exploration_blur)
print('press q to exit')
cv2.waitKey(100000) & 0xFF == ord('q')
print('done')

# fig, ax = plt.subplots(figsize=(10,10))
# plt.plot(exploration_blur)
# # load background
# background = plt.imread(bcgd_path)
# print('background loaded')
# # Add ROI
# plt.imshow(background, cmap='gray')
# ROI = patches.Rectangle((345,860),550,200,alpha=0.2,linewidth=3,edgecolor='r',facecolor='r')
# ax.add_patch(ROI)
# print('added ROI')

plt.show()