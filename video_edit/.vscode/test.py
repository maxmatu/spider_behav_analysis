import numpy as np
import pandas as pd
from nptdms import TdmsFile
from matplotlib import pyplot as plt
import sys
sys.path.append("C:\\Users\\maxma\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\spider_behav_analysis\\video_edit")
from LDR import get_stimuli 

tdms_path = 'C:\\Users\\maxma\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_escape\\06_03_20_sp8\\spider_camera_ldr(0).tdms' 
photodiode_raw_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_raw.png'
photodiode_save_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_smothened.png'
fps = 40
sampling_rate = 25000
stimuli = get_stimuli(tdms_path, photodiode_save_path, fps, sampling_rate, 8)
