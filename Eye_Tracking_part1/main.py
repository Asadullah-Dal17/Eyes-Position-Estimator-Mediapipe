import cv2 as cv 
import time 
# variables 
frame_counter =0
fonts = cv.FONT_HERSHEY_COMPLEX
# camera object
camera = cv.VideoCapture(0)
starting_time =time.time()
while True:
    frame_counter +=1
    ret, frame = camera.read()
    if not ret:
        break
    # frame =cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    end_time= time.time()-starting_time
    fps = frame_counter/end_time
    cv.putText(frame, f'FPS: {round(fps,2)}', (20,50), fonts, 1.2, (0,255, 0), 2)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
