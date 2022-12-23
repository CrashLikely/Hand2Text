import cv2
import time
import mediapipe as mp
import pandas as pd
import os

class HandTracker:
    def __init__(self,numhands=1):
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = False, max_num_hands = numhands,min_detection_confidence =0.5,min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0
        self.cTime = 0
        self.landmarks = []
    def track(self):
        while True:
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h,w,c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                        self.landmarks.append([cx,cy])
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
            self.cTime = time.time()
            fps = 1/(self.cTime-self.pTime)
            self.pTime=self.cTime
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.imshow("image",img)
            cv2.waitKey(1)
    def record(self):
        while True:
            self.landmarks = []
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h,w,c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                        self.landmarks.append([cx,cy])
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
            self.cTime = time.time()
            fps = 1/(self.cTime-self.pTime)
            self.pTime=self.cTime
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.imshow("image",img)
            k = cv2.waitKey(0)
            if k == 78:
                cv2.destroyWindow("Image")
            if k == 27:
                cv2.destroyAllWindows()

                try:
                    value = input("What is the value of these coordinates? ")
                except:
                    print(type(value))
                savePose = {"value":value,"landmarks":[self.landmarks]}
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
