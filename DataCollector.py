import cv2,time,os
import mediapipe as mp
import pandas as pd
import numpy as np
from HandChecker import HandChecker
from DataManager import DataManager
class DataCollector:
    def __init__(self,confidence):

        self.HC = HandChecker("files/saved_data.pickle",.00001,"categorical_crossentropy","categorical_accuracy")
        self.HC.CreateModel(True)
        self.DM = DataManager("files/saved_data.pickle",confidence)
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = False,max_num_hands=1,min_detection_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def Core(self):
        success, img = self.cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        landmarks=[]
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx,cy,cz = int(lm.x *w), int(lm.y * h), int(lm.z*w)
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                    landmarks.append([cx,cy,cz])
                self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        cv2.imshow("Image",img)
        k = cv2.waitKey(0)
        if k == 78:
            cv2.destroyWindow("Image")
            return "None"
        if k == 27:
            cv2.destroyAllWindows()
            try:
                value = input("What is the value of these coordinates?")
            except:
                print(type(value))
            savePose = {"value":value,"landmarks":[landmarks]}
            return savePose
        else:
            return "None"
    
    def CollectFromFiles(self,directory):
        '''
        This one is kinda useless tbh
        '''
        while True:
            for filename in os.listdire(directory):
                f = os.path.join(directory,filename)
                if os.path.isfile(f):
                    img = cv2.imread(f)
                    results = self.hands.process(img)
                    landmarks=[]
                    if results.multi_hand_landmarks:
                        for handLms in results.multi_hand_landmarks:
                            for id, lm in enumerate(handLms.landmark):
                                h,w,c = img.shape
                                cx,cy,cz = int(lm.x *w), int(lm.y *h), int(lm.z*w)
                                cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                                landmarks.append([cx,cy,cz])
                            self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
            cv2.imshow("Image",img)
            k = cv2.waitKey(0)
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
        
    def CollectDataFromCamera(self,path):
        '''
        Need to write this one, use this class instead of tutorial.py for collecting data
        '''
        while True:
            savePose = self.Core()
            if(savePose != "None"):
                if(os.path.exists(path)):
                    df = pd.read_pickle(path)
                    print(df.head)
                    df2 = pd.DataFrame(savePose)
                    print(df2.head)
                    df = pd.concat([df,df2],ignore_index=True)
                    df.to_pickle(path)
                else:
                    try:
                        df = pd.DataFrame(savePose)
                    except ValueError:
                        print(type(savePose),savePose)
                    print(df.head())
                    df.to_pickle(path)

    def TestData(self):
        '''
        Immediate testing of the data 
        '''
        while True:
            savePose = self.Core()
            if(savePose != "None"):
                model = self.HC.LoadModel()
                y_pred = model.predict(self.DM.ProcessLandmarks(savePose["landmarks"]))
                value = self.DM.ProcessValue(savePose["value"])
                print(y_pred)
                print(value)
                print(self.DM.GetGreatest(y_pred))

    def LiveFeed(self,visible):
        y_pred="Hand Not Detected"
        letter =""
        message=""
        while True:
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            landmarks=[]
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h,w,c = img.shape
                        cx,cy,cz = int(lm.x *w), int(lm.y * h), int(lm.z*w)
                        if(visible):
                            cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                        landmarks.append([cx,cy,cz])
                    if(visible):
                        self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
                model = self.HC.LoadModel()
                landmarks=np.asarray(landmarks)
                landmarks=landmarks.reshape(-1,21,3)
                y_pred = model.predict(self.DM.ProcessLandmarks(landmarks))
                y_pred,confidence = self.DM.GetGreatest(y_pred)
                confidence = f"Confidenc: {confidence}"
                if y_pred:
                    if letter != y_pred:
                        letter = y_pred
                        message += letter
                        if (len(message)>15):
                            message =""
                    cv2.putText(img,y_pred,(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
                    cv2.putText(img,confidence,(120,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
                else:
                    letter = "!"
                cv2.putText(img,message,(10,200),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
            cv2.imshow("Image",img)
            cv2.waitKey(100)
            


if __name__=="__main__":
    DC = DataCollector(96)
    #DC.CollectDataFromCamera("files/validation_data.pickle")
    #DC.TestData()
    DC.LiveFeed(True)
