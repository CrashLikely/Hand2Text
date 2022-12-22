import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from DataManager import DataManager


class HandChecker:
    def __init__(self,path,optimizer,loss,metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.DM = DataManager(path)
        self.DM.ScaleDown()
    
    def GatherTrainingData(self):
        self.data_values = self.DM.GetValues()
        self.data_values = keras.utils.to_categorical(self.data_values,26)
        self.data_landmarks = self.DM.GetLandmarks()
        print(type(self.data_values[0]))
        print(type(self.data_landmarks[0][0]))
        self.input_shape = self.data_landmarks.shape[1:]
        print(self.data_landmarks.shape)
        print(self.input_shape)
        self.output_shape = self.data_values.shape[1]
        print(self.data_values.shape)
        print(self.output_shape)

    def CreateModel(self):
        self.GatherTrainingData()
        self.model = keras.Sequential(name="HandPoseChecker")
        self.model.add(keras.layers.Dense(100,activation="relu",name="layer1",input_shape=(self.input_shape)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(26,activation="relu",name="layer2"))
        self.model.add(keras.layers.Dense(self.output_shape,name="layer3"))
        self.model.compile(optimizer = self.optimizer,
            loss=self.loss,  
            metrics = [self.metrics])
        self.model_loc = 'models/model.ckpt'
        self.model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath = self.model_loc,
            save_weights_only = True,
            monitor = self.metrics,
            mode = 'max',
            save_best_only=True)
        print(type(self.model_checkpoint_callback))
    
    def TrainModel(self):
        print(self.data_landmarks.shape)
        print(self.data_values.shape)
        history = self.model.fit(
            self.data_landmarks,
            self.data_values,
            batch_size=50,
            epochs=500,
            callbacks=[self.model_checkpoint_callback]
        )
        self.model.summary()
        plt.plot(history.history['categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

    def LoadModel(self):
        self.model.load_weights(self.model_loc)
        return self.model
if __name__=="__main__":
    HC = HandChecker("files/saved_data.pickle","Adam","categorical_crossentropy","categorical_accuracy")
    HC.CreateModel()
    HC.TrainModel()



