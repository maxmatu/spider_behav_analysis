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


videofilepath = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Raw_data\def_behav_probe\30.10.19_sp9.avi'
manual_video_inspect(videofilepath)
