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
# import matplotlib.pyplot as plt # for mac
import os.path
from keras.models import model_from_json
import PIL.ImageOps    
import sys
import time
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K 

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

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
 
        model.load_weights(os.path.join(f_model,weights_filename))

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        for name, layer in layer_dict:
            print("name:"+name)
        

if __name__ == "__main__":
    predictModel = predictDigit()
    predictModel.predict()
