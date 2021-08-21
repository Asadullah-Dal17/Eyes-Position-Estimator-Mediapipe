import cv2 as cv 
import time
import mediapipe as mp
import utils
# variables 
frame_counter =0
fonts = cv.FONT_HERSHEY_COMPLEX
map_face_mesh = mp.solutions.face_mesh

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

# landmarks detection function
def landmarksDetector(img, results, draw=False):
    img_height, img_width =img.shape[:2]
    mesh_coord = [(int(point.x*img_width),int(point.y*img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord
# camera object
camera = cv.VideoCapture('VideoFile.mp4')
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    starting_time =time.time()
    while True:
        frame_counter +=1
        ret, frame = camera.read()
        if not ret:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print('landmarks detect')
            mesh_coordinates= landmarksDetector(frame, results, False)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in FACE_OVAL], utils.GRAY, 0.7)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in RIGHT_EYE], utils.YELLOW, 0.3)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in LEFT_EYE], utils.YELLOW, 0.3)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in RIGHT_EYEBROW], utils.GREEN, 0.3)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in LEFT_EYEBROW], utils.GREEN, 0.3)
            frame = utils.fillPolyTrans(frame,[mesh_coordinates[p]for p in LIPS], utils.WHITE, 0.3)

            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,0), -1) for p in FACE_OVAL]
            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,0), -1) for p in RIGHT_EYE]
            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,0), -1) for p in LEFT_EYE]
            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,0), -1) for p in LIPS]
            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,255), -1) for p in LEFT_EYEBROW]
            # [cv.circle(frame, mesh_coordinates[p], 2, (0,255,255), -1) for p in RIGHT_EYEBROW]

        # frame =cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        end_time= time.time()-starting_time
        fps = frame_counter/end_time
        cv.putText(frame, f'FPS: {round(fps,2)}', (20,50), fonts, 1.2, (0,255, 0), 2)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
