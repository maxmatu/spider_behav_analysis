import numpy as np
import pandas as pd
from nptdms import TdmsFile
from matplotlib import pyplot as plt

tdms_path = 'C:\\Users\\maxma\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_escape\\06_03_20_sp8\\spider_camera_ldr(0).tdms' 
photodiode_raw_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_raw.png'
photodiode_save_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_smothened.png'
fps = 40
sampling_rate = 25000

#Load TDMS file
tdms_file = TdmsFile(tdms_path)
#Plot save and show Photodiode channel
photodiode_raw = tdms_file.group_channels('Photodiode')[0].data

onsets = np.where(np.diff(photodiode_raw) > .5)[0]
signal_duration = (8*sampling_rate)
stimuli = [onsets[0]]
for i in onsets:
     if i - stimuli[-1] > signal_duration:
          stimuli.append(i)
print(stimuli)
f, ax = plt.subplots()
ax.plot(photodiode_raw, color="r", lw=1)
signal = np.zeros(len(photodiode_raw))
for i in stimuli: 
    signal[i] = 1 
ax.plot(signal)
plt.show()
plt.savefig(photodiode_save_path)




