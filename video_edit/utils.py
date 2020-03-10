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

def remove_pulses(data, fps, sampling_rate):
    photodiode_smoothed = data.copy()
    signal_onset = np.where(np.diff(photodiode_smoothed) > .8)[0]
    signal_duration = (5*sampling_rate)
    for onset in signal_onset: 
        photodiode_smoothed[onset:onset+signal_duration] = photodiode_smoothed[onset]
    return photodiode_smoothed

def get_times_signal_high_and_low(signal, th=1):
    """
        Given a 1d time series it returns the times 
        (in # samples) in which the signal goes low->high (onset)
        and high->low (offset)

        :param signal: 1d numpy array or list with time series data
        :param th: float, the time derivative of signal is thresholded to find onset and offset
    """
    signal_copy = np.zeros_like(signal)
    signal_copy[signal > th] = 1

    signal_onset = np.where(np.diff(signal_copy) > .5)[0]
    signal_offset = np.where(np.diff(signal_copy) < -.5)[0]
    return signal_onset, signal_offset

def get_video_params(cap):
    if isinstance(cap, str):
        cap = cv2.VideoCapture(cap)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    return nframes, width, height, fps

def open_cvwriter(filepath, w=None, h=None, framerate=None, format='.mp4', iscolor=False):
    try:
        if 'avi' in format:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # (*'MP4V')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
    except:
        raise ValueError('Could not create videowriter')
    else:
        return videowriter
        
def trim_clip(videofilepath, savepath, start_frame=None, stop_frame=None):
    """trim_clip [take a videofilepath, open it and save a trimmed version between start and stop. Either 
    looking at a proportion of video (e.g. second half) or at start and stop frames]
    Arguments:
        videofilepath {[str]} -- [video to process]
        savepath {[str]} -- [where to save]
    Keyword Arguments:
        start_frame {[type]} -- [video frame to stat at ] (default: {None})
        stop_frame {[type]} -- [videoframe to stop at ] (default: {None})
    """
    # Open reader and writer
    cap = cv2.VideoCapture(videofilepath)
    nframes, width, height, fps  = get_video_params(cap)
    writer = open_cvwriter(savepath, w=width, h=height, framerate=int(fps), format='.mp4', iscolor=False)
    # Loop over frames and save the ones that matter
    print('Processing: ', videofilepath)
    cur_frame = 0
    cap.set(1,start_frame)
    while True:
        cur_frame += 1
        if cur_frame % 1000 == 0: print('Current frame: ', cur_frame)
        if cur_frame <= start_frame: continue
        elif cur_frame >= stop_frame: break
        else:
            ret, frame = cap.read()
            if not ret: break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(frame)
    writer.release()

def make_clips_session(videofilepath, onsets_frames, offsets_frames, s_before_start=5, s_after_end=5, fps=40):

    fld, name = os.path.split(videofilepath)
    
    new_folder = 'clips'
    new_folder_path = os.path.join(fld, new_folder)
    if not os.path.isdir(new_folder_path):
        os.makedirs(new_folder_path)
    print(" new folder created")

    for clipn, (start, end) in enumerate(zip(onsets_frames, offsets_frames)):
        clip_name, extention = name.split(".")[0], name.split(".")[1]
        print("Saving clip for stim {} of {}".format(clipn, len(onsets_frames)))
        clip_name += "_trial{}.{}".format(clipn, extention)
        
        clipfile = os.path.join(new_folder_path, clip_name)

        trim_clip(videofilepath, clipfile, start_frame=int(start)-fps*s_before_start, stop_frame=int(end)+fps*s_after_end)
    #shutil.move(tdms_path, new_folder_path)
    #shutil.move(videofilepath, new_folder_path)