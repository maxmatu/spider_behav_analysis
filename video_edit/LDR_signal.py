import numpy as np
import pandas as pd
from nptdms import TdmsFile
from matplotlib import pyplot as plt
from utils import remove_pulses

tdms_path = 'C:\\Users\\maxma\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_escape\\06_03_20_sp8\\spider_camera_ldr(0).tdms' 
photodiode_raw_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_raw.png'
photodiode_smoothed_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_smothened.png'
fps = 40
sampling_rate = 25000

#Load TDMS file
tdms_file = TdmsFile(tdms_path)
#Plot save and show Photodiode channel
photodiode_raw = tdms_file.group_channels('Photodiode')[0].data
plt.plot(photodiode_raw)
plt.savefig(photodiode_raw_path)
plt.show()

photodiode_smoothed, signal_onset = remove_pulses(photodiode_raw, fps, sampling_rate)
f, ax = plt.subplots()
ax.plot(photodiode_smoothed, color="r", lw=1)
signal = np.zeros(len(photodiode_smoothed))
for i in signal_onset: 
    signal[i] = 1 
ax.plot(signal)
plt.show()