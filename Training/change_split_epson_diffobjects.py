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

#folder = 'epson_split_time_models_cropped'

folder = 'epson_split_time_models_cropped2' #Different objects in val and test than in train

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

X = np.concatenate((X_train,X_valid,X_test))
Y = np.concatenate((Y_train,Y_valid,Y_test))
labels = np.concatenate((labels_train,labels_valid,labels_test))

ind = np.argsort(labels[:,0])

X = X[ind]
labels = labels[ind]
Y = Y[ind]

meshes = labels[:,0]
whats = np.unique(meshes)

train_ = np.argmax(meshes == whats[10])
valid_ = np.argmax(meshes == whats[13])

Y_train = Y[:train_]
X_train = X[:train_]
labels_train = labels[:train_]

Y_valid = Y[train_:valid_]
X_valid = X[train_:valid_]
labels_valid = labels[train_:valid_]

Y_test = Y[valid_:]
X_test = X[valid_:]
labels_test = labels[valid_:]

print(np.shape(X_train))
print(np.shape(X_valid))
print(np.shape(X_test))

np.save('epson_split_time_models_cropped3/X_train.npy',X_train)
np.save('epson_split_time_models_cropped3/Y_train.npy',Y_train)
np.save('epson_split_time_models_cropped3/labels_train.npy',labels_train)

np.save('epson_split_time_models_cropped3/X_valid.npy',X_valid)
np.save('epson_split_time_models_cropped3/Y_valid.npy',Y_valid)
np.save('epson_split_time_models_cropped3/labels_valid.npy',labels_valid)

np.save('epson_split_time_models_cropped3/X_test.npy',X_test)
np.save('epson_split_time_models_cropped3/Y_test.npy',Y_test)
np.save('epson_split_time_models_cropped3/labels_test.npy',labels_test)

np.save('epson_split_time_models_cropped3/X_views.npy',X_views)
np.save('epson_split_time_models_cropped3/Y_multiview.npy',Y_multiview)
np.save('epson_split_time_models_cropped3/labels_views.npy',labels_views)

