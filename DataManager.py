import pandas as pd
import numpy as np
from tensorflow import keras
class DataManager:
    def __init__(self,path,confidence=0.8):
        self.path = path
        self.confidence=confidence
        self.df = pd.read_pickle(path)
    def Save(self):
        self.df.to_pickle(self.path)
        print("Data saved")
    def PrintData(self,num_vals):
        for i in range(len(self.df)):
            if(not num_vals):
                print(f'value: {self.df["value"][i]} landmarks:{self.df["landmarks"][i]}')
                print("")
            else:
                print(f'value:{self.df["value"][i]} num_value"{self.df["num_value"][i]}')
    def WriteToFile(self,num_vals=True):
        f = open("2D_vald_values.txt","a")
        for i in range(len(self.df)):
            f.write(f"index: {i} value {self.df['value'][i]} num_value{self.df['num_value'][i]}\n")
        f.close()
    def Scale(self,value):
        return value/26.0
    def ScaleDown(self):
        '''
        Be careful when using this function that the data isn't already scaled down because then you would scale it down further. 
        '''
        for i in range(len(self.df)):
            self.df["num_value"][i] = (self.df["num_value"][i]/26.0)
    def merge(self,otherpath):
        other = pd.read_pickle(otherpath)
        print(other.head)
        print(self.df.head)
        new = pd.concat([self.df,other])
        self.df = new
    def ManualEdit(self,column,find,change):
        indexes=[]
        for i in range(len(self.df)):
            if(self.df[column][i]==find):
                self.df[column][i]=change
                indexes.append(i)
        print(f'indexes:{indexes} changed from {find} to {change}')
    def DeleteIndex(self,index):
        self.df.drop(index)
        print(f"{index} dropped")
    def ManualEditFromIndex(self,index,column,change):
        print(f"Changing {self.df[column][index]} to {change} at {index}")
        self.df[column][index]=change
    def GetLandmarks(self):
        '''
        Return training landmarks
        '''
        return np.asarray(self.df["landmarks"].values.tolist())
    def ProcessLandmarks(self,landmarks):
        return np.asarray(landmarks)

    def ProcessValue(self,value):
        '''
        Takes letter value of pose and returns a numerical value
        '''
        if value == 'A':value=1
        if value == 'B':value=2
        if value == 'C':value=3
        if value == 'D':value=4
        if value == 'E':value=5
        if value == 'F':value=6
        if value == 'G':value=7
        if value == 'H':value=8
        if value == 'I':value=9
        if value == 'J':value=10
        if value == 'K':value=11
        if value == 'L':value=12
        if value == 'M':value=13
        if value == 'N':value=14
        if value == 'O':value=15
        if value == 'P':value=16
        if value == 'Q':value=17
        if value == 'R':value=18
        if value == 'S':value=19
        if value == 'T':value=20
        if value == 'U':value=21
        if value == 'V':value=22
        if value == 'W':value=23
        if value == 'X':value=24
        if value == 'Y':value=25
        if value == 'Z':value=26
        return self.Scale(value)
    
    def CreateValueArray(self,value):
        value = round(value*26)
        value = value - 1
        array = np.zeros([26])
        array[value]=1.0
        return array
    def CreateNumValues(self):
        self.df["num_value"]=np.zeros(len(self.df))
        for i in range(len(self.df)):
            try:
                self.df['num_value'][i]=self.ProcessValue(self.df["value"][i])
            except TypeError as e:
                print(f"Error at index {i}. ")
                #print(self.df["value"][i])
                print(e)

    def GetValues(self):
        '''
        Return training Values
        '''
        self.CreateNumValues()
        temp = np.asarray(self.df["num_value"].values.tolist())
        values = []
        for i in range(len(temp)):
            values.append(self.CreateValueArray(temp[i]))
        return np.asarray(values)
    
    def getConfidence(self,prediction):
        confidences=[]
        total = 0
        greatest = 0
        for i in range(len(prediction[0])):
            total += prediction[0][i]
        for i in range(len(prediction[0])):
            confidences.append((prediction[0][i]/total))
        
        for confidence in confidences:
            differences = []
            for i in range(len(prediction[0])):
                differences.append(confidence-(prediction[0][i]/total))
            diff_sum = sum(differences)
            if (100-(abs((diff_sum/len(differences)))*100))>greatest:
                greatest = (100-(abs(diff_sum/len(differences))*100))
        return greatest


    def GetGreatest(self,prediction):
        greatest = -1000
        index = 0
        for i in range(len(prediction[0])):
            if(prediction[0][i]>greatest):
                greatest = prediction[0][i]
                index = i
        confidence = self.getConfidence(prediction)
        
        if(confidence>self.confidence):
            if index==0:return("A",confidence)
            if index==1:return("B",confidence)
            if index==2:return("C",confidence)
            if index==3:return("D",confidence)
            if index==4:return("E",confidence)
            if index==5:return("F",confidence)
            if index==6:return("G",confidence)
            if index==7:return("H",confidence)
            if index==8:return("I",confidence)
            if index==9:return("J",confidence)
            if index==10:return("K",confidence)
            if index==11:return("L",confidence)
            if index==12:return("M",confidence)
            if index==13:return("N",confidence)
            if index==14:return("O",confidence)
            if index==15:return("P",confidence)
            if index==16:return("Q",confidence)
            if index==17:return("R",confidence)
            if index==18:return("S",confidence)
            if index==19:return("T",confidence)
            if index==20:return("U",confidence)
            if index==21:return("V",confidence)
            if index==22:return("W",confidence)
            if index==23:return("X",confidence)
            if index==24:return("Y",confidence)
            if index==25:return("Z",confidence)
        else:
            print(f"Not enough confidence: {confidence}")
            return(False,confidence)
    def GetGreatestIndex(self,prediction):
        greatest = -1000
        index = 0
        for i in range(prediction.size):
            if(prediction[i] > greatest):
                greatest = prediction[i]
                index = i
        return index


if __name__=="__main__":
  print("Nice")
