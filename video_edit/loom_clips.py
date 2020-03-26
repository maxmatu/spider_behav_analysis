import numpy as np
from nptdms import TdmsFile
import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.signal import butter, lfilter, freqz, resample, wiener, gaussian
from scipy.ndimage import filters
from collections import namedtuple
import cv2  
import os
import sys
import shutil
from utils import remove_pulses
from utils import get_times_signal_high_and_low
from utils import make_clips_session

tdms_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\spider_camera_ldr(0).tdms' 
videofilepath = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\video.avi' 
photodiode_raw_path = r'C:\Users\maxma\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\06_03_20_sp8\photodiode_raw.svg'
fps = 40 
sampling_rate = 25000
# Load the tdms File with LDR signal
tdms_file = TdmsFile(tdms_path)

photodiode_raw = tdms_file.group_channels('Photodiode')[0].data
plt.plot(photodiode_raw)
plt.savefig(photodiode_raw_path)
plt.show()

photodiode_smoothed = remove_pulses(photodiode_raw, fps, sampling_rate)



onsets, offsets = get_times_signal_high_and_low(photodiode_smoothed, 0.8)
print(onsets, offsets)
print("Found {} stimuli".format(len(onsets)))
f, ax = plt.subplots()
ax.plot(photodiode_smoothed, color="r", lw=1)
plt.show()

#Converting a list of sampling timestamps to frame numbers
onsets_frames = np.int32((np.array(onsets) / sampling_rate) * fps)
offsets_frames = np.int32((np.array(offsets) / sampling_rate) * fps)
print(onsets_frames, offsets_frames)

#Cutting out and saving short clips of behaviour following the stimulus from the raw video
make_clips_session(videofilepath, onsets_frames, offsets_frames, s_before_start=5, s_after_end=10, fps=fps)
