import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import compute_distance
from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d as get_speed_from_xy

save_path = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\combined_speedplot.svg' 

trial_1 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial0DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
trial_2 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial1DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
trial_3 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial2DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
trial_4 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial3DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
trial_5 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial4DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
trial_6 = pd.read_hdf(r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\17_02_20_sp2\video_trial6DLC_resnet50_large_spidersJan27shuffle1_1030000.h5')
#trial_7 
#trial_8
x1, y1 = (trial_1['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_1['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])
x2, y2 = (trial_2['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_2['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])
x3, y3 = (trial_3['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_3['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])
x4, y4 = (trial_4['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_4['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])
x5, y5 = (trial_5['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_5['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])
x6, y6 = (trial_6['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['x'], trial_6['DLC_resnet50_large_spidersJan27shuffle1_1030000']['centre']['y'])

speed1 = get_speed_from_xy(x1, y1)
speed2 = get_speed_from_xy(x2, y2)
speed3 = get_speed_from_xy(x3, y3)
speed4 = get_speed_from_xy(x4, y4)
speed5 = get_speed_from_xy(x5, y5)
speed6 = get_speed_from_xy(x6, y6)
#plt.plot(speed1, speed2, speed3, speed4, speed5, speed6)
plt.plot(speed1, lw=0.5)
plt.plot(speed2, lw=0.5)
plt.plot(speed3, lw=0.5)
plt.plot(speed4, lw=0.5)
plt.plot(speed5, lw=0.5)
plt.plot(speed6, lw=0.5)
plt.savefig(save_path)
print('Plot saved')
plt.show()

