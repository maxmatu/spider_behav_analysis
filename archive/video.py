from nptdms import TdmsFile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, resample, wiener, gaussian
from scipy.ndimage import filters
from collections import namedtuple
import cv2  
import os
import sys

videofilepath = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Analysis\DLC_data\29.10.19_sp1DLC_resnet50_spider_analysisNov1shuffle1_1030000_labeled.mp4'

def manual_video_inspect(videofilepath):
        """[loads a video and lets the user select manually which frames to show]
                Arguments:
                        videofilepath {[str]} -- [path to video to be opened]
                key bindings:
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frame
                        - f: save frame
        """        
        def get_selected_frame(cap, show_frame):
                cap.set(1, show_frame)
                ret, frame = cap.read() # read the first frame
                return frame
                # Open text file to save selected frames
        fold, name = os.path.split(videofilepath)
        frames_file = open(os.path.join(fold, name.split('.')[0])+".txt","w+")
        cap = cv2.VideoCapture(videofilepath)
        if not cap.isOpened():
                raise FileNotFoundError('Couldnt load the file')
        print(""" Instructions
                        - d: advance to next frame
                        - .: advance 40 frames/1s
                        - a: go back to previous frame
                        - ,: go back 40 frames/1s
                        - s: select frame
                        - f: save frame number
                        - q: quit
        """)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Initialise showing the first frame
        show_frame = 18134.0
        frame = get_selected_frame(cap, show_frame)
        while True:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(25)
                if k == ord('d'):
                        # Display next frame
                        if show_frame < number_of_frames:
                                show_frame += 1
                elif k == ord('.'):
                        # Display next frame
                        if show_frame < number_of_frames:
                                show_frame += 40
                
                elif k == ord('a'):
                        # Display the previous frame
                        if show_frame > 1:
                                show_frame -= 1
                elif k == ord(','):
                        # Display the previous frame
                        if show_frame > 1:
                                show_frame -= 40
                elif k ==ord('s'):
                        selected_frame = int(input('Enter frame number: '))
                        if selected_frame > number_of_frames or selected_frame < 0:
                                print(selected_frame, ' is an invalid option')
                        show_frame = int(selected_frame)
                elif k == ord('f'): 
                    print('Saving frame to text')
                    frames_file.write('\n'+str(show_frame))
                elif k == ord('q'):
                    frames_file.close()
                    sys.exit()
                try:
                        frame = get_selected_frame(cap, show_frame)
                        print('Showing frame {} of {}'.format(show_frame, number_of_frames))
                except:
                        raise ValueError('Could not display frame ', show_frame)

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
        
def trim_clip(videopath, savepath, 
                start_frame=None, stop_frame=None):
    """trim_clip [take a videopath, open it and save a trimmed version between start and stop. Either 
    looking at a proportion of video (e.g. second half) or at start and stop frames]
    Arguments:
        videopath {[str]} -- [video to process]
        savepath {[str]} -- [where to save]
    Keyword Arguments:
        start_frame {[type]} -- [video frame to stat at ] (default: {None})
        end_frame {[type]} -- [videoframe to stop at ] (default: {None})
    """
    # Open reader and writer
    cap = cv2.VideoCapture(videopath)
    nframes, width, height, fps  = get_video_params(cap)
    writer = open_cvwriter(savepath, w=width, h=height, framerate=int(fps), format='.mp4', iscolor=False)
    # Loop over frames and save the ones that matter
    print('Processing: ', videopath)
    cur_frame = 0
    cap.set(1,start_frame)
    while True:
        cur_frame += 1
        if cur_frame % 100 == 0: print('Current frame: ', cur_frame)
        if cur_frame <= start_frame: continue
        elif cur_frame >= stop_frame: break
        else:
            ret, frame = cap.read()
            if not ret: break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(frame)
    writer.release()

manual_video_inspect(videofilepath)