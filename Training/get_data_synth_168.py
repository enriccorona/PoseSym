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


split = 1

folder = 'synth_80_split_' + str(split)

#X_train = np.load(folder + '/X_train.npy')
rgb_train = np.load(folder + '/rgb_train.npy')
Y_train = np.load(folder + '/Y_train.npy')
labels_train = np.load(folder + '/labels_train.npy')

#X_valid = np.load(folder + '/X_valid.npy')
rgb_valid = np.load(folder + '/rgb_valid.npy')
Y_valid = np.load(folder + '/Y_valid.npy')
labels_valid = np.load(folder + '/labels_valid.npy')

#X_test = np.load(folder + '/X_test.npy')
rgb_test = np.load(folder + '/rgb_test.npy')
Y_test = np.load(folder + '/Y_test.npy')
labels_test = np.load(folder + '/labels_test.npy')

X_views = np.load(folder + '/X_views.npy')
#X_views = np.load(folder + '/X_views.npy')
Y_multiview = np.load(folder + '/Y_multiview.npy')
labels_views = np.load(folder + '/labels_views.npy')

# No need to test model on synthetic data, so let's add data to train
#X_train = np.concatenate((X_train,X_test))
rgb_train = np.concatenate((rgb_train,rgb_test))
Y_train = np.concatenate((Y_train,Y_test))
labels_train = np.concatenate((labels_train,labels_test))

#print(X_train.shape)

# SOMETHING WEIRD IN OBJECTS: 3726, 865, 390, 379, 4052. REMOVING THEM:
num_to_conserve = np.ones(len(labels_train[:,0]))
num_to_conserve[labels_train[:,0] == 3726] = 0
num_to_conserve[labels_train[:,0] == 865] = 0
num_to_conserve[labels_train[:,0] == 390] = 0
num_to_conserve[labels_train[:,0] == 379] = 0
num_to_conserve[labels_train[:,0] == 4052] = 0

#X_train = X_train[num_to_conserve]
rgb_train = rgb_train[num_to_conserve==1]
Y_train = Y_train[num_to_conserve==1]
labels_train = labels_train[num_to_conserve==1]

print(X_views.max())

def load_all_dataset():
  return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views

def load_dataset():
  return 1,Y_train,rgb_train,labels_train,1,Y_valid,rgb_valid,labels_valid,X_views,Y_multiview,labels_views
  #return X_train,Y_train,rgb_train,labels_train,X_valid,Y_valid,rgb_valid,labels_valid,X_views,Y_multiview,labels_views
  #return X_train,Y_train,labels_train,X_valid,Y_valid,labels_valid,X_views,Y_multiview,labels_views

def load_test_dataset():
  return X_test,Y_test,labels_test,X_views,Y_multiview,labels_views

def load_all_dataset_to_recompute_groundtruth():
  X_train = np.load(folder + '/rgb_train.npy')
  Y_train = np.load(folder + '/Y_train.npy')
  labels_train = np.load(folder + '/labels_train.npy')

  X_valid = np.load(folder + '/rgb_valid.npy')
  Y_valid = np.load(folder + '/Y_valid.npy')
  labels_valid = np.load(folder + '/labels_valid.npy')

  X_test = np.load(folder + '/rgb_test.npy')
  Y_test = np.load(folder + '/Y_test.npy')
  labels_test = np.load(folder + '/labels_test.npy')

  X_views = np.load(folder + '/X_views.npy')
  Y_multiview = np.load(folder + '/Y_multiview.npy')
  labels_views = np.load(folder + '/labels_views.npy')

  print("Y is: ")
  print(Y_train.shape)

  X_train = np.concatenate((X_train,X_valid,X_test))
  Y_train = np.concatenate((Y_train,Y_valid,Y_test))
  labels_train = np.concatenate((labels_train,labels_valid,labels_test))

  X_train = X_train[num_to_conserve]
  Y_train = Y_train[num_to_conserve]
  labels_train = labels_train[num_to_conserve]
  
  return X_train,Y_train,labels_train,X_views,Y_multiview,labels_views
