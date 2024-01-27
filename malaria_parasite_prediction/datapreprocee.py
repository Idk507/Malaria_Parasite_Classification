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
allImages = []
allLabels = []

input_shape = (124,124)

parasite_path = r'C:\Users\danus\Downloads\malaria\cell_images\Parasitized'
uninfected_path = r'C:\Users\danus\Downloads\malaria\cell_images\Uninfected'

paths = [parasite_path, uninfected_path]

for path in paths:
    path2 = path + "/*.png"
    for file in glob.glob(path2):
        print(file)
        #load the images
        
        img = cv2.imread(file)
        
        if img is not None:
            resized = cv2.resize(img,input_shape,interpolation = cv2.INTER_AREA)
            allImages.append(resized)
            
            if path == parasite_path:
                allLabels.append(1)
            else:
                allLabels.append(0)   
            
allImagesNP = np.array(allImages)
print(allImagesNP.shape)

allLabelsNP = np.array(allLabels)
print(allLabelsNP.shape)

np.save("c:/Users/danus/Downloads/malaria/allImagesNP.npy", allImagesNP)
np.save("c:/Users/danus/Downloads/malaria/allLabelsNP.npy", allLabelsNP)

