import cv2
import time
import mediapipe as mp
import pandas as pd
import os


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence =0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    landmarks=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                if(id not in range(0,2) and id !=5 and id !=9 and id !=13 and id!=17):
                    cv2.putText(img,f"x:{cx},y{cy}",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                    print(f"id:{id}, x:{cx},y:{cy}")
                landmarks.append([cx,cy])
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    k = cv2.waitKey(0) 
    if k == 78:
        cv2.destroyWindow("Image")
    if k == 27:
        cv2.destroyAllWindows()
        try:
            value = input("What is the value of these coordinates?")
        except:
            print(type(value))
        savePose = {"value":value,"landmarks":[landmarks]}
        print(len(landmarks))
        if(os.path.exists("files/saved_data.pickle")):
            df = pd.read_pickle("files/saved_data.pickle"
            )
            print(df.head)
            df2 = pd.DataFrame(savePose)
            print(df2.head)
            df = pd.concat([df,df2],ignore_index=True)
            df.to_pickle('files/saved_data.pickle')
        else:
            df = pd.DataFrame(savePose)
            print(df.head())
            df.to_pickle('files/saved_data.pickle')
        
        