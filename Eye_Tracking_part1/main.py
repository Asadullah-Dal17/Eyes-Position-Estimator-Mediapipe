import cv2 as cv 
import time 
import utils
import mediapipe as mp
# varables 
frame_counter =0
font = cv.FONT_HERSHEY_COMPLEX

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

# face mesh object 
map_face_mesh = mp.solutions.face_mesh
# functions 

# landmarks detector function 
def landmarksDetector(img, results, draw=False)->list:
    # getting image height and width 
    img_height, img_width =img.shape[:2]
    #looping through normalized landmarks, converting the to img coordinates
    mesh_marks_coord = [(int(point.x*img_width), int(point.y*img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        # draw circle on each landmark 
        [cv.circle(img, mark_point, 2, utils.GREEN, -2) for mark_point in mesh_marks_coord]
    return mesh_marks_coord

# camera object 
camera = cv.VideoCapture('VideoFile.mp4')
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # strating time here 
    starting_time = time.time()
    # loop through frames 
    while True:
        frame_counter+=1
        ret, frame = camera.read()
        if not ret:
            break 
        # converting BGR frame to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # getting the landmarks
        results = face_mesh.process(rgb_frame)
        # checking if landmarks are detected or not first
        if results.multi_face_landmarks:
            # print(f'landmarks_detected_{frame_counter}')
            # calling landmark detection function
            mesh_coord = landmarksDetector(frame, results=results, draw=False)
            # draw transparent shape on each part, like eyes, lips, face_oval, and eyebrows 
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in FACE_OVAL], utils.GRAY, 0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in LIPS], utils.PINK, 0.2)
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in LEFT_EYE], utils.BLUE, 0.2)
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in LEFT_EYEBROW], utils.GREEN, 0.2)
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in RIGHT_EYE], utils.BLUE, 0.2)
            frame = utils.fillPolyTrans(frame, [mesh_coord[point] for point in RIGHT_EYEBROW], utils.GREEN, 0.2)




        end_time = time.time()-starting_time
        fps = frame_counter/end_time
        cv.putText(frame, f'FPS: {round(fps,2)}', (20,50), font, 1.0, utils.GREEN, 2)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key ==ord('q') or key==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()