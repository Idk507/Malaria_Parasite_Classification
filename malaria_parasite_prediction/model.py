import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import io
import sklearn
import pandas as pd 
import glob
import cv2 
import time
import random
from sklearn.metrics import roc_curve, confusion_matrix
from PIL import Image
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.callbacks import ModelCheckpoint
#build a CNN model

allImages = np.load("c:/Users/danus/Downloads/malaria/allImagesNP.npy")
allLabels = np.load("c:/Users/danus/Downloads/malaria/allLabelsNP.npy")
input_shape = (124,124,3)
shpae = (124,124)
#first image

img = allImages[0]
label = allLabels[0]

plt.imshow(img)
plt.show()

#prepare all the data

allImagesForModel = allImages / 255.0

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(allImagesForModel,allLabels,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

model = Sequential()

model.add(Conv2D(input_shape=input_shape,filters=16,kernel_size=(3,3),padding="same",activation="relu"))
model.add(Conv2D(filters = 16,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(1024,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
batch = 32
epochs = 10

stepsperepoch = np.ceil(len(x_train)/batch)
validationsteps = np.ceil(len(x_test)/batch)

model_file = "c:/Users/danus/Downloads/malaria/malaria_model.h5"

bmodel = ModelCheckpoint(model_file,monitor="val_accuracy",verbose=1,save_best_only=True)
history = model.fit(x_train,y_train,batch_size=batch,epochs=epochs,validation_data=(x_test,y_test),callbacks=[bmodel],steps_per_epoch=stepsperepoch,validation_steps=validationsteps)