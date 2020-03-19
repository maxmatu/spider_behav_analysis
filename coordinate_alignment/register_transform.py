from Video_Functions import register_arena, get_background
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
from roi_analysis import get_timeinrois_stats

#Naming convention:
# videopath = 'Z:\swc\branco\007Max\deeplabcut\videos\videos_10_02_20\06_02_20_sp13.mp4' 
# ~~~~~~~~~~~~~~~ CARE FOR .AVI OR .MP4 !!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bcgd_path = r'C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_exploration\\06_02_20_sp13_background.png'
# save_path = r'C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_exploration\\06_02_20_sp13_transform.npy'
video_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14\video.avi'
bcgd_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14_background.png'
save_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\vertical_arena_exploration\06_03_20_sp14_transform.npy'

def correct_tracking_data(uncorrected, M):
    # Do the correction 
    m3d = np.append(M, np.zeros((1,3)),0)
    corrected = np.zeros_like(uncorrected)

    # affine transform to match model arena
    concat = np.ones((len(uncorrected), 3))
    concat[:, :2] = uncorrected
    corrected = np.matmul(m3d, concat.T).T[:, :2]
    return corrected

def register_transform(video_path,background_save=False, background_path, ):
    
    # Get transform matrix
    background_image = get_background(video_path, start_frame=0, avg_over=10)
    print('finding bacground')
    # Register transform
    cv2.imwrite(bcgd_path, background_image)
    M, _, _, _  = register_arena(background_image, None, 0, 0, show_arena = False)
    
    # Save transform matrix
    np.save(save_path, M)



