import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

data = pd.read_pickle("files/saved_data.pickle")

data_values = data["value"]
data_landmarks = data["landmarks"] 



print(data_landmarks.shape)
print("DATA TYPES BABY!!!!")
print(type(data_landmarks))
print(type(data_landmarks[0]))
print(data_landmarks[0].shape)
print(type(data_landmarks[0][0]))
print(data_landmarks[0][0].shape)
#Model
model = keras.Sequential(name="HandPoseChecker")
model.add(layers.Flatten())
model.add(layers.Dense(2,activation="relu", name="layer1"))
model.add(layers.Dense(3,activation="relu",name="layer2"))
model.add(layers.Dense(4,name="layer3"))

#x = tf.ones(1,21,2)
#y = model(x)

model.compile(optimizer = keras.optimizers.Adam(),
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics =[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(
    data_landmarks,
    data_values,
    batch_size=50,
    epochs=10
)