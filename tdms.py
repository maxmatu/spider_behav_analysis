from nptdms import TdmsFile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.signal import butter, lfilter, freqz, resample, wiener, gaussian
from scipy.ndimage import filters
from collections import namedtuple

tdms_path = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Raw_data\def_behav_probe\30_10_19_sp10_LDR.tdms'
tdms_file = TdmsFile(tdms_path)

photodiode_raw = tdms_file.group_channels('Photodiode')[0].data
spider_camera_input = tdms_file.group_channels('spider_camera_input')[0].data
low_pulse_indexes = np.where(photodiode_raw < 0.1)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_noise(data):
    low_pulse_indexes = np.where(data < 0.1)[0]
    photodiode_smoothed = data.copy()
    photodiode_smoothed[low_pulse_indexes] = photodiode_smoothed[low_pulse_indexes-1]

    return photodiode_smoothed

def find_peaks_in_signal(signal, time_limit, th, above=True):
    """[Function to find the start of square peaks in a time series. 
    Useful for example to find frame starts or stim starts in analog input data]
    Arguments:
        signal {[np.array]} -- [the time series to be analysd]
        time_limit {[float]} -- [min time inbetween peaks]
        th {[float]} -- [where to threshold the signal to identify the peaks]
    Returns:
        [np.ndarray] -- [peak starts times]
    """
    if above:
        above_th = np.where(signal>th)[0]
    else:
        above_th = np.where(signal<th)[0]
    if not np.any(above_th): return np.array([])
    peak_starts = [x for x,d in zip(above_th, np.diff(above_th)) if d > time_limit]
    # add the first and last above_th times to make sure all frames are included
    peak_starts.insert(0, above_th[0])
    peak_starts.append(above_th[-1])
    # we then remove the second item because it corresponds to the end of the first peak
    peak_starts.pop(1)
    return np.array(peak_starts)

def find_visual_stimuli(data, th, sampling_rate):
    """ [Filter the data to remove high freq noise, then take the diff and thereshold to find changes]
        Arguments:
            data {[np.ndarray]} -- [1D numpy array with sample data (extracted from TDMS)]
            th {[int]} -- [Threshold]
            sampling_rate {[int]} -- [sampling rate of experiment]
    """

    filtered  = butter_lowpass_filter(remove_noise(data), 75, int(sampling_rate/2))
    d_filt = np.diff(filtered)
    # find start and ends of stimuli (when it goes above and under threhsold)
    ends = find_peaks_in_signal(d_filt, 1, -0.0025, above=False )
    starts = find_peaks_in_signal(d_filt, 10, 0.0025, above=True )[1:]

    # ignore ends picked up at the very end of signal
    ends = [e for e in ends if len(data)-e > 500000]

    # if the number of starts and ends doesnt match something went wrong
    if not len(starts) == len(ends):
        if abs(len(starts)-len(ends))>1: raise ValueError("Too large error during detection: s:{} e{}".format(len(starts), len(ends)))
        print("Something went wrong: {} - starts and {} - ends".format(len(starts), len(ends)))
        # ? Fo1r debugging
        f, ax = plt.subplots()
        ax.plot(filtered, color='r')
        ax.plot(butter_lowpass_filter(np.diff(filtered), 75, int(sampling_rate/2)), color='g')
        ax.scatter(starts, [0.25 for i in starts], c='r')
        ax.scatter(ends, [0 for i in ends], c='k')
        plt.show()
        to_elim = int(input("Which one to delete "))
        if len(starts)  > len(ends):
            starts = np.delete(starts, to_elim)
        else:
            ends = np.delete(ends, to_elim)

    assert len(starts) == len(ends), "cacca"

    # Return as a list of named tuples
    stim = namedtuple("stim", "start end")
    stimuli =[stim(s,e) for s,e in zip(starts, ends)]
    for s,e in stimuli:  # check that the end is after the start
       if e < s: raise ValueError("Wrong stimuli detection")
    return stimuli, filtered

stim_times, filtered = find_visual_stimuli(photodiode_raw, 1.5, 25000)
print(stim_times)
print("Found {} stimuli".format(len(stim_times)))

print("plotting")
f, ax = plt.subplots()
ax.plot(filtered, color="r", lw=1)
for start, end in stim_times:
    ax.axvline(start, color="g")
    ax.axvline(end, color="b")

plt.show()




    

