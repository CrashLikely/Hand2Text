import pandas as pd
import numpy as np
from tensorflow import keras
class DataManager:
    def __init__(self,path):
        self.df = pd.read_pickle(path)
    def PrintData(self):
        print(self.df)
    def Scale(self,value):
        return value/26.0
    def ScaleDown(self):
        '''
        Be careful when using this function that the data isn't already scaled down because then you would scale it down further. 
        '''
        for i in range(len(self.df)):
            self.df["num_value"][i] = (self.df["num_value"][i]/26.0)
    
    def GetLandmarks(self):
        '''
        Return training landmarks
        '''
        return np.asarray(self.df["landmarks"].values.tolist())
    def ProcessLandmarks(self,savePose):
        return np.asarray(savePose["landmarks"])

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
        value = int(value*26)
        value = value - 1
        array = np.zeros([26])
        array[value]=1.0
        return array

    def GetValues(self):
        '''
        Return training Values
        '''
        temp = np.asarray(self.df["num_value"].values.tolist())
        values = []
        for i in range(len(temp)):
            values.append(self.CreateValueArray(temp[i]))
        return np.asarray(values)
    def GetGreatest(self,prediction):
        greatest = -1000
        index = 0
        for i in range(len(prediction[0])):
            if(prediction[0][i]>greatest):
                greatest = prediction[0][i]
                index = i
        if index==0:return("A")
        if index==1:return("B")
        if index==2:return("C")
        if index==3:return("D")
        if index==4:return("E")
        if index==5:return("F")
        if index==6:return("G")
        if index==7:return("H")
        if index==8:return("I")
        if index==9:return("J")
        if index==10:return("K")
        if index==11:return("L")
        if index==12:return("M")
        if index==13:return("N")
        if index==14:return("O")
        if index==15:return("P")
        if index==16:return("Q")
        if index==17:return("R")
        if index==18:return("S")
        if index==19:return("T")
        if index==20:return("U")
        if index==21:return("V")
        if index==22:return("W")
        if index==23:return("X")
        if index==24:return("Y")
        if index==25:return("Z")



if __name__=="__main__":
    dc = DataManager("files/saved_data.pickle")
    dc.ScaleDown()
    values = dc.GetValues()
    print(values[8])
    dc.PrintData()
