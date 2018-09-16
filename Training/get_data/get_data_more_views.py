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
#import matplotlib.pyplot as plt

img_size = 224/2 #224 for vgg

def load_views():
#  multiview_folder = '/ais/gobi5/enricproject/data_cnn/multiview_depth80/'
  multiview_folder = '/ais/gobi5/enricproject/data_cnn/multiview_more/'

  X_views = []
  Y_multiview = []
  labels_views = []
  
  models_todo = np.array([134,2922,2980,135,5481,223,224,2960] + [4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012,4014,4015,4016,4017,4018,4019] + [5009,5010,5011] )

  models = glob.glob(multiview_folder + 'Cropped2_model_*.npy')
#  models2 = glob.glob(multiview_folder + 'Cropped2_model_50[0-1]*.npy')
  #for i in range(len(models2)):
  #  models.append(models2[i])

#  print(models)

  models.sort()
  cameras = glob.glob(multiview_folder + 'RMat*')

  for i in range(len(cameras)):
    filename = str.split(cameras[i],'/')[-1]
    camera = str.split(str.split(filename,'cam_')[1],'.')[0]
    Y_multiview.append(np.load(multiview_folder + 'RMat_cam_' + str(i+1) + '.npy' ) )

  for i in range(len(models)):
    filename = str.split(models[i],'/')[-1]
    print(filename)
    mesh = str.split(str.split(filename,'model_')[1],'_')[0]
    if int(mesh) in models_todo:
    #if True:
      print(int(mesh))
      camera = str.split(str.split(filename,'cam_')[1],'.')[0]
      labels_views.append( np.array( [int(mesh),int(camera) ] ) )
      img = np.load(models[i])
      #print(img.shape)
      #img = cv2.resize( img, (img_size,img_size) )
      #img = (img - np.mean(img))/np.std(img)
      #img[img!=0] = img[img!=0] - img[img!=0].mean()
      #img = img/np.std(img)
      X_views.append( img )
      #X_views.append( cv2.resize( np.load(models[i]), (img_size,img_size) ) )

  X_views = np.array(X_views)
  labels_views = np.array(labels_views)
  
  Y_multiview = np.array(Y_multiview)
  print(np.shape(labels_views))
  #X_views = X_views/np.max(X_views)

  return X_views,Y_multiview,labels_views

