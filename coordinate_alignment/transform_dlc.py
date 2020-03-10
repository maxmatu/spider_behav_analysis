import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from register_transform import correct_tracking_data
from Video_Functions import model_arena

def get_scorer_from_dataframe(df):
    first_frame = df.iloc[0]
    bodyparts = first_frame.index.levels[1]
    scorer = first_frame.index.levels[0]
    return scorer, bodyparts

    
if __name__ == "__main__":
    # After DLC and you have the transform matrix saved somewhere...
    M = np.load('C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\Analysis\\confined_shade_arena_exploration\\06_02_20_sp13_transform.npy')

    # Load DLC
    pose_file = 'C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\Analysis\\confined_shade_arena_exploration\\06_02_20_sp13_trimmed_tracking.h5'
    posedata = pd.read_hdf(pose_file)
    corrected_posedata = posedata.copy()

    # Transform tracking and save
    scorer, bodyparts = get_scorer_from_dataframe(posedata)
    for bp in bodyparts:
        # Get XY pose and correct with CCM matrix
        xy = posedata[scorer[0], bp].values[:, :2]
        corrected_data = correct_tracking_data(xy, M)
        corrected_posedata[scorer[0], bp].values[:, :2] = corrected_data

    plt.plot(posedata[scorer[0], bp].values[:, 0]) # raw data X coord
    plt.plot(corrected_posedata[scorer[0], bp].values[:, 0]) # transformed data X coord
    plt.show()
    corrected_posedata.to_hdf('C:\\Users\\maksymilianm\\Dropbox (UCL - SWC)\\Project_spiders\\Analysis\\confined_shade_arena_exploration\\06_02_20_sp13_aligned_tracking.h5', 'hdf')

