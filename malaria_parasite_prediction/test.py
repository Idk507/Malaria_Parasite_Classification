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

#Testing the model 

model_file = "c:/Users/danus/Downloads/malaria/malaria_model.h5"

model = tf.keras.models.load_model(model_file)
input_shape = (124,124)

categories = ["Parasitized","Uninfected"]

def prepare_img(img):
    resized = cv2.resize(img,input_shape,interpolation = cv2.INTER_AREA)
    imgresult = np.expand_dims(resized,axis=0)
    imgresult = imgresult/255
    
    return imgresult

testimage = r"cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png"

img = cv2.imread(testimage)

imgformodel = prepare_img(img)

result = model.predict(imgformodel)

print(result)
    
    