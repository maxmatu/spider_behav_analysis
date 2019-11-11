import numpy as np
start_time = 11334033
sampling_rate = 25000
fps = 40 
frame_number = np.round(np.multiply(np.divide(start_time, sampling_rate), fps))
print('frame', frame_number)
print('time:',frame_number/40/60, 'min')