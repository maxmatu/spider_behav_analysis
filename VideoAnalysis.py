import cv2 
import numpy as np 

cap = cv2.VideoCapture(r'Z:\swc\branco\\007Max\Data\def_behav_probe\\17.10.19_sp1\\17.10.19_sp1.avi')

i=0 #i is an integer corresponding to current frame number
#playing = True 
#while(playing):
while(cap.isOpened()): 
    
    ret, frame = cap.read()     
    if ret == False: 
        break
    cv2.imshow('frame', frame)

    key = cv2.waitKey(0)
    while key not in [ord('q'), ord('k')]:
        key = cv2.waitKey(0)
    # Quit when 'q' is pressed
    if key == ord('q'):
        break
    
    #while key not in [ord('q'), ord('k')]:
    #    key = cv2.waitKey(0)

    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    break 

    #if cv2.waitKey(10) & 0xFF == ord('n'):
    #    playing = False  
    #if cv2.imshow() is done: 
     #  break
    print('Frame:', i)
    i+=1

# while True:
#     image = cap.read()
#     count = 0;
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#   if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#       break
#   count += 1 




# Display the video in a separate window. Press q to exit. 
# while True:

#     _, frame = cap.read()
#     cv2.imshow('display', frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break  




cap.release()
cv2.destroyAllWindows()