import cv2,time,os
import mediapipe as mp
import pandas as pd
from HandChecker import HandChecker
from DataManager import DataManager
class DataCollector:
    def __init__(self):

        self.HC = HandChecker("files/saved_data.pickle",.00001,"categorical_crossentropy",["categorical_accuracy"])
        self.HC.CreateModel()
        self.DM = DataManager("files/saved_data.pickle")

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
                    cx,cy = int(lm.x *w), int(lm.y * h)
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                    landmarks.append([cx,cy])
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
                                cx,cy = int(lm.x *w), int(lm.y *h)
                                cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
                                landmarks.append([cx,cy])
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
        
    def CollectDataFromCamera(self):
        '''
        Need to write this one, use this class instead of tutorial.py for collecting data
        '''
        while True:
            savePose = self.Core()
            if(os.path.exists("files/saved_data.pickle")):
                df = pd.read_pickle("files/saved_data.pickle")
                print(df.head)
                df2 = pd.DataFrame(savePose)
                print(df2.head)
                df = pd.concat([df,df2],ignore_index=True)
                df.to_pickle('files/saved_data.pickle')
            else:
                df = pd.DataFrame(savePose)
                print(df.head())
                df.to_pickle('files/saved_data.pickle')


    def TestData(self):
        '''
        Immediate testing of the data 
        '''
        while True:
            savePose = self.Core()
            if(savePose != "None"):
                model = self.HC.LoadModel()
                y_pred = model.predict(self.DM.ProcessLandmarks(savePose))
                value = self.DM.ProcessValue(savePose["value"])
                print(y_pred)
                print(value)
                print(self.DM.GetGreatest(y_pred))

if __name__=="__main__":
    DC = DataCollector()
    DC.TestData()