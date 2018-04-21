#!/usr/bin/env python

import glob
import numpy as np
#from __future__ import print_function
import scipy.io
import sys
import os
import time
import numpy
import cv2
from eval_pair_symm import * #eval_pair
#import matplotlib.pyplot as plt

img_size = 224/2 #224 for vgg

folder = 'epson_split_336' # timestamp
#folder = 'epson_split_168views_objects1322'
#folder = 'epson_split_168views_objects1034'
#folder = 'epson_split_168_models_cropped4'
#folder = 'epson_split_168views_objects1034_22'

#folder = 'epson_split_648' # timestamp
#folder = 'epson_split_648views_objects1322' 
#folder = 'epson_split_648views_objects1034' 

X_train = np.load(folder + '/X_train.npy')
Y_train = np.load(folder + '/Y_train.npy')
labels_train = np.load(folder + '/labels_train.npy')

X_valid = np.load(folder + '/X_valid.npy')
Y_valid = np.load(folder + '/Y_valid.npy')
labels_valid = np.load(folder + '/labels_valid.npy')

X_test = np.load(folder + '/X_test.npy')
Y_test = np.load(folder + '/Y_test.npy')
labels_test = np.load(folder + '/labels_test.npy')

X_views = np.load(folder + '/X_views.npy')
Y_multiview = np.load(folder + '/Y_multiview.npy')
labels_views = np.load(folder + '/labels_views.npy')

X_views = X_views*0.157089233398
print(X_views.max())

def load_all_dataset():
  return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views

def load_dataset():
  return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_views,Y_multiview,labels_views

def load_test_dataset():
  return X_test,Y_test,labels_test,X_views,Y_multiview,labels_views
