
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import csv
# デバッグ用csv出力
f = open('out.csv', 'w', newline='')
data = [['time','ratio']]

# variables 
frame_counter = 0 # FPS計算用
ratioList = [0 for i in range(10)] # 直近10フレーム分のeye_ratioを記憶．直近データはratioList[9].
CEF_COUNTER = 0 # 目を閉じていたフレームをカウント
PREB_COUNTER = 0 # 直前の瞬きからフレームをカウント
NON_FACE_COUNTER = 0 # 顔を検知しなかったフレーム数をカウント
TOTAL_BLINKS = 0 # 瞬きの回数0
# constants
CLOSED_EYES_FRAME = 1 # このフレーム数より多く目を閉じていればまばたきを検出
COOL_FRAME = 8 # 瞬きを連続で誤検出しないようにするためのクールタイム (フレーム)
FONTS =cv.FONT_HERSHEY_COMPLEX

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(1)
# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    # rv_top = landmarks[right_indices[12]]
    # rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line 
    # lv_top = landmarks[left_indices[12]]
    # lv_bottom = landmarks[left_indices[4]]

    rvDistances = []
    lvDistances = []
    rhDistance = euclaideanDistance(rh_right, rh_left)
    # rvDistance = euclaideanDistance(rv_top, rv_bottom)
    # lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    for i in range(7):
        rvDistances.append(euclaideanDistance(landmarks[right_indices[1+i]], landmarks[right_indices[15-i]]))
        lvDistances.append(euclaideanDistance(landmarks[left_indices[1+i]], landmarks[left_indices[15-i]]))
    
    # Eye Ratio の計算方法は独自に考案したものに変更した．
    # 目の縦の長さを7箇所測って合計したものを，目の横の長さ(目頭と目尻の距離)で割った値としている．
    reRatio = sum(rvDistances)/rhDistance
    leRatio = sum(lvDistances)/lhDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Return Detected or not 
def is_blinked(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9):
    thr = 0.2
    if l0-l2 > thr and l6-l2 > thr:
        return True
    if l0-l2 > thr and l7-l2 > thr:
        return True
    if l0-l2 > thr and l8-l2 > thr:
        return True
    if l0-l2 > thr and l9-l2 > thr:
        return True
    if l0-l3 > thr and l7-l3 > thr:
        return True
    if l0-l3 > thr and l8-l3 > thr:
        return True
    if l0-l3 > thr and l9-l3 > thr:
        return True
    return False


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            if NON_FACE_COUNTER > CLOSED_EYES_FRAME:
                if NON_FACE_COUNTER < 10 and PREB_COUNTER > COOL_FRAME:
                    TOTAL_BLINKS += 1
                    PREB_COUNTER = 0
                NON_FACE_COUNTER = 0
            else:
                mesh_coords = landmarksDetection(frame, results, False)
                for i in range(len(ratioList) - 1):
                    ratioList[i] = ratioList[i+1]
                ratioList[len(ratioList) - 1] = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratioList[0],2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

                if is_blinked(*ratioList) and PREB_COUNTER > COOL_FRAME:
                    CEF_COUNTER += 1
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                elif CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
                    PREB_COUNTER = 0

                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                # CEF_COUNTER +=1
            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
        else:
            utils.colorBackgroundText(frame,  f'Face is not Detected', FONTS, 1.7, (int(frame_height/2), 80), 2, utils.YELLOW, pad_x=6, pad_y=6, )
            NON_FACE_COUNTER += 1
        PREB_COUNTER += 1

        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        # デバッグ用データ
        data.append([end_time, ratioList[len(ratioList) - 1]])

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key==ord('Q') or key==27:
            break
    cv.destroyAllWindows()
    camera.release()

# デバッグ用csv出力
writer = csv.writer(f)
writer.writerows(data)
f.close()
