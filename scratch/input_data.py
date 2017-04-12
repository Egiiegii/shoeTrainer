import os
from PIL import Image
import PIL.ImageOps
# import urllib.parse # for python 3
from urlparse import urlparse # for python 2
from itertools import count
import random
import cv2
import numpy as np
from keras.datasets import mnist
import pandas as pd 
from keras.datasets import cifar10

"""
data/
    train/
        0/
            dog001.jpg
            dog002.jpg
            ...
        1/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        0/
            dog001.jpg
            dog002.jpg
            ...
        1/
            cat001.jpg
            cat002.jpg
            ...
"""

def make_label_for_each_directory(dir):
	"""
	:param dir
	:return (big_array, label_array): (nparray, label)
	"""

	big_array = []
	label_array = []
	class_label = 0
	for index, class_name in enumerate(os.listdir(dir), start=0):
		abs_path_class_name = dir + "/" + class_name
		if(os.path.isdir(abs_path_class_name)):
			print("labeling class_name"+'%s/%s'%(dir,str(class_name)) + "as " '%s'%class_label)
			labelname = class_label
			class_label = class_label + 1
			#appending index to label_array 
			for pic_name in os.listdir(abs_path_class_name):
				abs_path_pic_name = abs_path_class_name + "/" + pic_name
				if(verify(abs_path_pic_name)):
					pic_array = read_image_as_array(abs_path_pic_name)
					big_array.append(pic_array)
					# print(np.asarray(big_array).shape)
					label_array.append(labelname)
		else:
			print("not folder, %s" % class_name)
	#turn python array to numpy
	np_big_array = np.asarray(big_array)
	# ('y_train shape:', (120,))
	np_label_array = np.asarray(label_array).reshape(-1, 1)
	return (np_big_array, np_label_array)

def read_image_as_array(dir):
	"""
	:param dir: directory of image name
	:return array
	"""

	im = cv2.imread(dir)
	if im.shape[0] != 56 and im.shape[1] != 56:
		im = cv2.resize(im, (56, 56))
	im = im.tolist()
	return im

def egii_data_generate(data_directory="data"):
	"""
	:param cur_path: input directory of train, validation data 
	:return (x_train, y_train), (x_test, y_test)
	"""
	cur_path = os.path.dirname(os.path.abspath('__file__'))
	abs_path = cur_path + "/" + data_directory

	(x_train, y_train) = make_label_for_each_directory(abs_path + "/train")
	(x_test, y_test) = make_label_for_each_directory(abs_path + "/validation")
	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	return (x_train, y_train), (x_test, y_test)

def verify(img_name):
	"""
	:param img_name: 
	:return Verify if img_name is image
	"""
	try:
		im=Image.open(img_name)
		return True
	except IOError:
		return False


if __name__ == '__main__':
	data_directory = "data"
	(x_train, y_train), (x_test, y_test) = egii_data_generate(data_directory)

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	# print(x_train)
	# print(x_train.shape[1:])
	print(y_train)
	# print("saving to csv")

	# df = pd.DataFrame(x_train)
	# df.to_pickle(file_name)
	# df.to_csv("x_train"+".csv")

	# df = pd.DataFrame(y_train)
	# df.to_pickle(file_name)
	# df.to_csv("y_train"+".csv")

	# df = pd.DataFrame(x_test)
	# df.to_pickle(file_name)
	# df.to_csv("x_test"+".csv")

	# df = pd.DataFrame(y_test)
	# df.to_pickle(file_name)
	# df.to_csv("y_test"+".csv")

	# print("finished saving to csv")


	# df = pd.read_pickle(file_name)

	# The data, shuffled and split between train and test sets:


	# x_train shape: (50000, 32, 32, 3)
	# y_train shape: (50000, 1)
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	# print(x_train)
	# print(x_train.shape[1:])
	print(y_train)



# Reference
# http://stackoverflow.com/questions/29839350/numpy-append-vs-python-append