import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from DataManager import DataManager
from keras.optimizers import SGD
import time
class HandChecker:
    def __init__(self,path,learning,loss,metrics):
        self.optimizer = SGD(lr=learning)
        self.loss = loss
        self.metrics = metrics
        self.DM = DataManager(path)
        self.valdDM = DataManager("files/validation_data.pickle")
        self.dev=False
    
    def devMode(self,value):
        self.dev = value

    def GatherTrainingData(self):
        self.data_values = self.DM.GetValues()
        self.data_landmarks = self.DM.GetLandmarks()
        print(self.data_values[0])
        print(self.data_landmarks[0])
        self.input_shape = self.data_landmarks.shape[1:]
        print(self.data_landmarks.shape)
        print(self.input_shape)
        self.output_shape = self.data_values.shape[1]
        print(self.data_values.shape)
        print(self.output_shape)

    def GatherValidationData(self):
        self.vald_values = self.valdDM.GetValues()
        self.vald_landmarks = self.valdDM.GetLandmarks()

    def CreateModel(self,checkpoint):
        self.checkpoint = checkpoint
        self.GatherTrainingData()
        self.GatherValidationData()
        self.model = keras.Sequential(name="HandPoseChecker")
        self.model.add(keras.layers.Dense(63,activation="relu",name="layer1",input_shape=(self.input_shape)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.output_shape,activation="sigmoid",name="layer3"))
        self.model.compile(optimizer = self.optimizer,
            loss=self.loss,  
            metrics = [self.metrics])
        if checkpoint:
            self.model_loc = f'models/model.ckpt'
            self.model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath = self.model_loc,
                save_weights_only = True,
                monitor = "val_"+self.metrics,
                mode = 'max',
                save_best_only=True)
        #print(type(self.model_checkpoint_callback))
    
    def TrainModel(self,epochs,batch_size):
        print(self.data_landmarks.shape)
        print(self.data_values.shape)
        if self.checkpoint:
            history = self.model.fit(
                self.data_landmarks,
                self.data_values,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.vald_landmarks,self.vald_values),
                callbacks=[self.model_checkpoint_callback]
            )
        else:
            history = self.model.fit(
                self.data_landmarks,
                self.data_values,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.vald_landmarks,self.vald_values)
            )
        self.model.summary()
        if(self.dev):
            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.show()
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()

    def LoadModel(self):
        self.model.load_weights(self.model_loc).expect_partial()
        return self.model

    def ValidateModel(self):
        '''
        This method tests the model on the validation dataset and keeps track of bias in guesses. 
        Should also keep track of Correct Guesses vs False Ones and at the indexes that those occur (to see if it is better at guess some rather than others).
        '''
        false = np.zeros(26)
        true = np.zeros(26)
        sum_false=0.0
        sum_true=0.0
        print(len(self.vald_landmarks))
        print(self.vald_landmarks[0])
        print(self.vald_values[0])
        print(self.vald_landmarks[0].shape)
        print(len(self.vald_landmarks))
        for i in range(len(self.vald_landmarks)):
            landmarks = self.vald_landmarks[i].reshape(-1,21,3)
            y_pred = self.model.predict(landmarks)
            pred_index = self.DM.GetGreatestIndex(y_pred[0])
            true_index = self.DM.GetGreatestIndex(self.vald_values[i])
            if pred_index==true_index:
                true[true_index]+=1
                sum_true+=1.0
            else:
                false[true_index]+=1
                sum_false+=1.0
        alphabet=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        data_true={}
        data_false={}
        for i in range(len(alphabet)):
            print(f"{alphabet[i]}-- false:{false[i]} true:{true[i]}")
            data_true[f"{alphabet[i]}"]=true[i]
            data_false[f"{alphabet[i]}"]=false[i]
        true_labels = list(data_true.keys())
        false_labels = list(data_false.keys())
        true_values = list(data_true.values())
        false_values = list(data_false.values())
        
        print(f"Overall Validation score: {sum_true/(sum_true+sum_false)}")
        if(self.dev):
            plt.bar(true_labels,true_values,color="green",width=0.8)
            plt.bar(false_labels,false_values,color="red",width=0.4)
            plt.show()
        
        return(sum_true/(sum_true+sum_false))
    
if __name__=="__main__":
    HC = HandChecker("files/saved_data.pickle",0.0001,"categorical_crossentropy","categorical_accuracy")
    #
    #HC.GatherTrainingData()
    HC.devMode(True)
    HC.CreateModel(True)
    HC.TrainModel(750,100)
    HC.LoadModel()
    HC.ValidateModel()
    # start = time.time()
    # epochs=[5,25,50,75,100]
    # averages=[]
    # times=[]
    # num=0
    # for epoch in epochs:
    #     temp_sum=0.0
    #     time_sum=0.0
    #     for i in range(5):
    #         HC.CreateModel(False)
    #         start_time=time.time()
    #         HC.TrainModel(500,epoch)
    #         #HC.LoadModel()
    #         temp_sum+=HC.ValidateModel()
    #         temp_time=time.time()-start_time
    #         time_sum+=temp_time
    #     averages.append(temp_sum/5.0)
    #     times.append(time_sum/5)
    # print(averages)
    # print(times)
    # print(f"total time: {time.time()-start}")