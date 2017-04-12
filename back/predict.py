#coding: utf-8
'''
Kizu Recog 2017.1.13
 
'''
from __future__ import print_function
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt
import os.path
from keras.models import model_from_json
import PIL.ImageOps    
import sys
import time
from PIL import Image, ImageDraw, ImageFont
 
np.random.seed(1337)  # for reproducibility
 
class predictDigit(object):
    def predict(self):
        """
        input:
            filename
        output:
            accuracy
        """

        size = (56, 56)
        argvs = sys.argv
        argc = len(argvs) 
        # print(argvs)
        # print(argc)

        # input image name
        fileName=argvs[1]
 
        winW=56 #sliding window width
        winH=56 #sliding window height
 
        f_log = './log'
        f_model = './model'
        model_filename = 'cnn_model.json'
        weights_filename = 'cnn_model_weights.hdf5'
        batch_size = 55
 
        # 1. load model
        json_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_json(json_string)
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
 
        #set compiler
        model.compile(loss='mean_squared_error',
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.load_weights(os.path.join(f_model,weights_filename))
        
        #2. load picture
        im = Image.open(fileName, 'r')
        
        if im is None:
            print("no image file selected")
            return
 
        # im = PIL.ImageOps.invert(im)
        im.save("temp.jpg")
        im=cv2.imread("temp.jpg")
        im= np.array(im)
        forShowIm = cv2.imread(fileName)
        # if the window does not meet our desired window size, ignore it
        forShowIm = cv2.resize(forShowIm, size)

        cv2.imwrite('01.png',forShowIm)    
        windowNP = np.asarray(forShowIm)
        windowNP= windowNP.reshape(-1,56,56,3)

        # 5. for each window, predict foreign object certainty
        preds = model.predict(windowNP, batch_size=26, verbose=1)
        
        print(preds)
        


predictModel = predictDigit()
predictModel.predict()
