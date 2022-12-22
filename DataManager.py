import pandas as pd
import numpy as np

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
        return np.asarray([savePose["landmarks"]])

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
    def GetValues(self):
        '''
        Return training Values
        '''
        return np.asarray(self.df["num_value"].values.tolist())
    


if __name__=="__main__":
    dc = DataManager("files/saved_data.pickle")
    dc.ScaleDown()
    dc.PrintData()
