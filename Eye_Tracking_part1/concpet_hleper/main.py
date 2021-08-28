import cv2 as cv
import mediapipe as mp
import time
import utils
import numpy as np

# variables 
frame_counter =0

# constants 
FONTS =cv.FONT_HERSHEY_COMPLEX
# Landmarks indices for different parts of face, mediapipe.

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# camera object 
camera = cv.VideoCapture('VideoFile.mp4')

# function
# landmarks detector fucntion 
def landmarkDetection(img, results, draw=False):
    # getting image width and height
    img_height, img_width = img.shape[:2]
    # looping through landmarks, converting normalized landmarks to pixel coordinates list[tuples(x,y)]
    print("\n")
    mesh_coords = [(point.x, point.y) for point in results.multi_face_landmarks[0].landmark]
    print(mesh_coords[:3])
    print("")
# 
    mesh_coords = [(int(point.x*img_width), int(point.y*img_height)) for point in results.multi_face_landmarks[0].landmark]
    print(mesh_coords[:3])
    print()

    if draw:
        # draw circle on each landmarks 
        [cv.circle(img, point, 2, utils.GREEN, -1) for point in mesh_coords]
    # mask = np.zeros((900, 900, 3), dtype=np.uint8)
    # mesh_coords1 = [(int(point.x*900), int(point.y*900)) for point in results.multi_face_landmarks[0].landmark]

    # [cv.circle(mask, point, 2, utils.GREEN, -1) for point in mesh_coords1]
    # cv.imshow('mask', mask)


    return mesh_coords # list of tuples(x,y)
# setting up mediapipe 
map_face_mesh = mp.solutions.face_mesh

# config mediapipe 
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()

    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        # converting color space of frame 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # resized =cv.resize(rgb_frame, None, fx=0.1, fy=0.1, interpolation=cv.INTER_CUBIC)
        # cv.imshow('resized', resized)
        # cv.putText(frame, f'input: {resized.shape[:2]} out: {frame.shape[:2]}', (20, 100), FONTS, 0.6, utils.GREEN, 2 )



        results = face_mesh.process(rgb_frame)
        # checking for landmarks 
        if results.multi_face_landmarks:
            mesh_coord = landmarkDetection(frame, results, False)
            # print(mesh_coord[:3])
            print("")
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in FACE_OVAL], utils.GRAY, 0.6)
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in LEFT_EYE], utils.GREEN, 0.3)
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in LEFT_EYEBROW], utils.YELLOW, 0.3)
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in RIGHT_EYE], utils.GREEN, 0.3)
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in RIGHT_EYEBROW], utils.YELLOW, 0.3)
            # frame = utils.fillPolyTrans(frame, [mesh_coord[p] for p in LIPS], utils.PINK, 0.3)


        # calc fps 
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50), FONTS, 1.0, utils.GREEN, 2 )

        # cv.imshow('frame', frame)
        key = cv.waitKey(0)
        break
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()