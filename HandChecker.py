import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_pickle("files/saved_data.pickle")

data_values = np.asarray(data["num_value"].values.tolist())
print(data_values.shape)
print(data_values[5])


data_landmarks = np.asarray(data.landmarks.values.tolist())
print(data_landmarks.shape)
print(type(data_landmarks))
print(type(data_values[0]))
#Model
model = keras.Sequential(name="HandPoseChecker")
model.add(layers.Dense(200,activation="relu", name="layer1"))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(layers.Dense(100,activation="relu",name="layer2"))
model.add(layers.Flatten())
model.add(layers.Dense(26,name="layer3"))

#x = tf.ones(1,21,2)
#y = model(x)

model.compile(optimizer = 'adam',
loss='MeanSquaredError',
metrics =['accuracy'])

history = model.fit(
    data_landmarks,
    data_values,
    batch_size=50,
    epochs=500
)
model.summary()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()