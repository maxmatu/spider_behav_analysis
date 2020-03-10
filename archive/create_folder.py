import os
import errno

videofilepath = r'C:\Users\maksymilianm\Dropbox (UCL - SWC)\Project_spiders\Raw_data\def_behav_probe\08.11.19_sp10.avi'
if not os.path.exists(os.path.dirname(videofilepath)):
    try:
        os.makedirs(os.path.dirname(videofilepath))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
#with open(videofilepath, "w") as f:
#    f.write("FOOBAR")