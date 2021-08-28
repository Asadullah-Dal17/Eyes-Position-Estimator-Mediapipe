import cv2 as cv
import mediapipe as mp
import time
import utils

# variables 
frame_counter =0

# constants 
FONTS =cv.FONT_HERSHEY_COMPLEX

# camera object 
camera = cv.VideoCapture("VideoFile.mp4")

start_time = time.time()

while True:
    frame_counter +=1 # frame counter
    ret, frame = camera.read() # getting frame from camera 
    if not ret: 
        break # no more frames break
    # converting color space of frame 
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # calculating  frame per seconds FPS

    end_time = time.time()-start_time
    fps = frame_counter/end_time
    cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50), FONTS, 1.0, utils.GREEN, 2 )

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key==ord('q') or key ==ord('Q'):
        break
cv.destroyAllWindows()
camera.release()
