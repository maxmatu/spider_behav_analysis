import numpy as np
import pandas as pd
from nptdms import TdmsFile
from matplotlib import pyplot as plt
from utils import remove_pulses

tdms_path = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\confined_shade_arena_escape\19_02_20_sp7\spider_camera_ldr(0).tdms' 
tdms_file = TdmsFile(tdms_path)
fps = 40
sampling_rate = 25000

photodiode_raw = tdms_file.group_channels('Photodiode')[0].data
photodiode_smoothed = remove_pulses(photodiode_raw, fps, sampling_rate)

f, ax = plt.subplots()
ax.plot(photodiode_smoothed, color="r", lw=1)
plt.show()