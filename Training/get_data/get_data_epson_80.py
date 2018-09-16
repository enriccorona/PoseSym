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
from utils.eval_pair_symm import * #eval_pair
#import matplotlib.pyplot as plt

img_size = 224/2 #224 for vgg

#folder = 'epson_split_time_models_cropped'

#folder = 'epson_split_time_models_cropped2' #Different objects in val and test than in train 13-2-2
folder = 'epson_split_time_models_cropped3' #Different objects in val and test than in train 10-3-4
#folder = 'epson_split_time_models_cropped4' #Different objects in val and test than in train 10-3-4
#folder = 'epson_split_time_models_cropped5' #Different objects in val and test than in train 10-3-4
#folder = 'epson_split_time_models_cropped6' #Different objects in val and test than in train 10-3-4

X_train = np.load(folder + '/X_train.npy')
Y_train = np.load(folder + '/Y_train.npy')
labels_train = np.load(folder + '/labels_train.npy')

X_valid = np.load(folder + '/X_valid.npy')
Y_valid = np.load(folder + '/Y_valid.npy')
labels_valid = np.load(folder + '/labels_valid.npy')

X_test = np.load(folder + '/X_test.npy')
Y_test = np.load(folder + '/Y_test.npy')
labels_test = np.load(folder + '/labels_test.npy')

#X_views = np.load(folder + '/X_views.npy')
#Y_multiview = np.load(folder + '/Y_multiview.npy')
#labels_views = np.load(folder + '/labels_views.npy')

X_views = np.load(folder + '/X_views2.npy')
Y_multiview = np.load(folder + '/Y_multiview2.npy')
labels_views = np.load(folder + '/labels_views2.npy')

if False:  
  m = np.max(X_views,1)
  m = np.max(m,1)
  print(m.shape)
  m = np.repeat(m,112*112)
  m = m.reshape([-1,112,112])
  X_views = X_views/m

  m = np.max(X_views,1)
  m = np.max(m,1)
  print(m[0:10])

  X_views = X_views/np.max(X_views)
  print(X_views.max())

print(X_views.max())



print(labels_train.shape)

def load_all_dataset():
  return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views

def load_dataset():
#  return X_train,Y_train,labels_train,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views
  return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_views,Y_multiview,labels_views

def load_valid_dataset():
  return X_valid,Y_valid,labels_valid,X_views,Y_multiview,labels_views

def load_test_dataset():
#  return X_valid,Y_valid,labels_valid,X_views,Y_multiview,labels_views
  return X_test,Y_test,labels_test,X_views,Y_multiview,labels_views
