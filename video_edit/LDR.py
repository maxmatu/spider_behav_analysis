import numpy as np
import pandas as pd
from nptdms import TdmsFile
from matplotlib import pyplot as plt

def get_stimuli(tdms_path, save_path, fps, sampling_rate, duration):

    #Load TDMS file
    tdms_file = TdmsFile(tdms_path)
    #Plot save and show Photodiode channel
    photodiode_raw = tdms_file.group_channels('Photodiode')[0].data

    onsets = np.where(np.diff(photodiode_raw) > .5)[0]
    signal_duration = (duration*sampling_rate)
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
    plt.savefig(save_path)
    return stimuli




