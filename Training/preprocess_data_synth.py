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
# Folder to save:
folder = 'split_scale'
using_symmetries_inferred = False

def load_dataset():
  scene_folder = []
  multiview_folder = []

  # Folders where scene images are:
  scene_folder.append('/ais/gobi5/enricproject/data_cnn/sceneview/')
  scene_folder.append('/ais/gobi5/enricproject/data_cnn/sceneview_all/')
  # Folders where object views are: (with the corresponding views)
  multiview_folder.append('/ais/gobi5/enricproject/data_cnn/multiview_depth80/')
  multiview_folder.append('/ais/gobi5/enricproject/data_cnn/multiview_more/')

  X_scene = []
  rgb_scene = []
  Y_scene = []
  labels_scene = []
  
  X_views = []
  Y_multiview = []
  labels_views = []

#  models_accepted = [28,29,198,199,704,1126,5008,6078]
  models_accepted = []
  
  scenes = glob.glob(scene_folder[0] + 'img_depth_*')#[0:50]
  scene_folder_num = [0]*len(scenes)
  scenes += glob.glob(scene_folder[1] + 'img_depth_*')#[0:50]
  scene_folder_num += [1]*(len(scenes) - len(scene_folder_num))
  for i in range(len(scenes)):
    filename = str.split(scenes[i],'/')[-1]
    mesh_name = str.split(str.split(filename,'img_depth_obj_')[1],'_')[0]
    mesh = int(float(str.split(str.split(filename,'img_depth_obj_')[1],'_')[0]))
    scene = str.split(str.split(filename,'scene_')[1],'_')[0]
    camera = str.split(str.split(filename,'cam_')[1],'.')[0]

    if int(mesh) not in models_accepted:
      models_accepted.append(int(mesh))

    if int(camera) != 1:#  and int(mesh) in models_accepted:
    #if int(mesh) != 203 and int(mesh) != 204 and int(mesh) != 120 and int(mesh) != 4548 and int(mesh) != 1126 and int(mesh) != 5100 and int(scene) > 600: # or mesh == 1126: #
      if (len(glob.glob(scene_folder[scene_folder_num[i]] + 'RMat_obj_' + str(mesh) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy') ) > 0):
        img = np.load( scenes[i] )
#        img[img!=0] = img[img!=0] - img[img!=0].mean()
#        img = img/np.std(img)
#        img = (img - np.mean(img))/np.std(img)
        size = np.shape(img)
        new_img = np.zeros([np.max(size),np.max(size)])
        if size[0] > size[1]:
          start = (size[0] - size[1])/2
          end = size[0] - (size[0] - size[1])/2 - 1
          if end-start < size[1]:
            end += 1
          elif end-start < size[1]:
            end -= 1

          new_img[:,start:end] = img
        else:
          start = (size[1] - size[0])/2
          end = size[1] - (size[1] - size[0])/2 
          if end - start < size[0]:
            end += 1
          elif end - start > size[0]:
            end -= 1
          new_img[start:end,:] = img
        if np.max(size) < 20:
          continue

        new_img_depth = cv2.resize(new_img,(img_size,img_size) )
        #if(len(glob.glob(scene_folder[scene_folder_num[i]] + 'img_obj_' + str(mesh_name) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy') ) > 0):
        #  img = np.load(scene_folder[scene_folder_num[i]] + 'img_obj_' + str(mesh_name) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy')
        if(len(glob.glob(scene_folder[scene_folder_num[i]] + 'img_cropped_obj_' + str(mesh_name) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy') ) > 0):
          print(scene_folder[scene_folder_num[i]] + 'img_cropped_obj_' + str(mesh_name) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy')
          try:
            img = np.load(scene_folder[scene_folder_num[i]] + 'img_cropped_obj_' + str(mesh_name) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy')
          except:
            continue
          size = np.shape(img)
          new_img = np.zeros([np.max(size),np.max(size),3])
          if np.max(size) < 20:
            continue
          if size[0] > size[1]:
            start = (size[0] - size[1])/2
            end = size[0] - (size[0] - size[1])/2 - 1
            if end-start < size[1]:
              end += 1
            elif end-start < size[1]:
              end -= 1

            new_img[:,start:end,:] = img
          else:
            start = (size[1] - size[0])/2
            end = size[1] - (size[1] - size[0])/2
            if end - start < size[0]:
              end += 1
            elif end - start > size[0]:
              end -= 1
            new_img[start:end,:,:] = img
          new_img = cv2.resize(new_img,(img_size,img_size) )

          rgb_scene.append(new_img)
        else:
          continue
          rgb_scene.append([])

        labels_scene.append( np.array( [int(mesh), int(scene), int(camera) ] ) )
        X_scene.append( new_img_depth )
        Y_scene.append(np.load(scene_folder[scene_folder_num[i]] + 'RMat_obj_' + str(mesh) + '_scene_' + str(scene) + '_cam_' + str(camera) + '.npy'))

  print(models_accepted)
  X_scene = np.array(X_scene)
  labels_scene = np.array(labels_scene)
  rgb_scene = np.array(rgb_scene)
  Y_scene = np.array(Y_scene)

 # models = glob.glob(multiview_folder + 'model_40[0-1]*.npy')
  #models = glob.glob(multiview_folder + 'model_*.npy')
  models = glob.glob(multiview_folder[0] + 'Cropped2_model_*.npy')
#  models_folder = [0]*len(models)

  models += glob.glob(multiview_folder[1] + 'Cropped2_model_*.npy')
#  models_folder += [1]*(len(models) - len(models_folder))
  
  cameras = glob.glob(multiview_folder[0] + 'RMat*')

  for i in range(len(cameras)):
    filename = str.split(cameras[i],'/')[-1]
    camera = str.split(str.split(filename,'cam_')[1],'.')[0]
    Y_multiview.append(np.load(multiview_folder[0] + 'RMat_cam_' + str(i+1) + '.npy' ) )

  for i in range(len(models)):
    filename = str.split(models[i],'/')[-1]
    mesh = str.split(str.split(filename,'model_')[1],'_')[0]
    camera = str.split(str.split(filename,'cam_')[1],'.')[0]

    if int(mesh) in models_accepted:

      labels_views.append( np.array( [int(mesh),int(camera) ] ) )
      img = np.load(models[i])#[:,210:750]
      #img = cv2.resize( img, (img_size,img_size) )
      #img = (img - np.mean(img))/np.std(img)
      #img[img!=0] = img[img!=0] - img[img!=0].mean()
      #img = img/np.std(img)
      X_views.append( img )
      #X_views.append( cv2.resize( np.load(models[i]), (img_size,img_size) ) )

  X_views = np.array(X_views)
  labels_views = np.array(labels_views)
  
  Y_multiview = np.array(Y_multiview)
  
  
  #ind = np.arange(len(X_scene))
  #np.random.shuffle(ind)
  ind = np.argsort(labels_scene[:,0])

# IF USING DIFFERENT OBJECTS FOR TRAIN THAT FOR TEST:
#  ind = np.argsort(labels_scene[:,0])

  X_scene = X_scene[ind]
  rgb_scene = rgb_scene[ind]
  labels_scene = labels_scene[ind]
  Y_scene = Y_scene[ind]

  print(labels_scene)
  print(labels_scene.shape)
  
  #print(np.shape(X_scene))
#  X_views = (X_views - np.mean(X_views))/np.std(X_views)
#  X_scene = (X_scene - np.mean(X_scene))/np.std(X_scene)
  #X_views = X_views/np.max(X_views)
  #X_scene = X_scene/np.max(X_scene)
  #print(np.min(X_views))
  #print(np.max(X_views))
  #print(np.min(X_scene))
  #print(np.max(X_scene))
  #print(np.shape(rgb_scene))

  return X_scene,Y_scene,rgb_scene,labels_scene,X_views,Y_multiview,labels_views


#if "__main__" == __name__:

_,Y_scene,X_scene,labels_scene,X_views,Y_multiview,labels_views = load_dataset()

from get_data_scale_3 import load_all_dataset_to_recompute_groundtruth

#X_scene,Y_scene,labels_scene,X_views,Y_multiview,labels_views = load_all_dataset_to_recompute_groundtruth()

if using_symmetries_inferred:
  symmetries_inferred = np.load('symmetries_inferred.npy')
  models_inferred = np.load('meshes_inferred.npy')

num_cams = np.max(labels_views[:,1])
#print(num_cams)
new_labels_scene = np.zeros([len(labels_scene), 3 + num_cams])
#print(labels_scene)

print("Size is " + str(len(X_scene)))
for index1 in range( len(X_scene) ): 
  mesh = labels_scene[index1,0]

  # RETRIEVE IN DATABASE RAW CONTAINING MESH
  nums_mesh = (labels_views[:,0] == mesh)
  lab = labels_views[nums_mesh]

  if len(lab) == 0:
    continue
  print(lab[0])

  if len(lab[0,2:5]) == 0:
    ind = (models_inferred == mesh)
    if using_symmetries_inferred:
      symmetry = symmetries_inferred[ind][0]
    else:
      symmetry = [1,1,1]
  else:
    if using_symmetries_inferred:
      symmetry = lab[0,2:5]
    else:
      symmetry = [1,1,1]

  new_labels_scene[index1,0] = labels_scene[index1,0]
  new_labels_scene[index1,1] = labels_scene[index1,1]
  new_labels_scene[index1,2] = labels_scene[index1,2]

  for index2 in range( len(lab) ):
    num_cam = int(lab[index2,1])
#    print(lab[0,2:5])
    if index2!=0 and symmetry[0] == np.Inf and symmetry[1] == np.Inf and symmetry[2] == np.Inf:
      new_labels_scene[index1, 3 + num_cam - 1] = new_labels_scene[index1, 3 + previous_num_cam - 1]
    else:
      new_labels_scene[index1, 3 + num_cam - 1] = eval_pair( Y_scene[index1], labels_scene[index1], Y_multiview[num_cam-1], lab[index2], symmetry )

    previous_num_cam = num_cam


labels_scene = np.array(new_labels_scene[:])

length = len(Y_scene)
print(length)
train_ = int(length*0.8)
valid_ = int(length*0.9)

Y_train = Y_scene[:train_]
X_train = X_scene[:train_]
labels_train = labels_scene[:train_]

Y_valid = Y_scene[train_:valid_]
X_valid = X_scene[train_:valid_]
labels_valid = labels_scene[train_:valid_]

Y_test = Y_scene[valid_:]
X_test = X_scene[valid_:]
labels_test = labels_scene[valid_:]

print(np.shape(X_train))
print(np.shape(X_valid))
print(np.shape(X_test))

np.save(folder + '/rgb_train.npy',X_train)
np.save(folder + '/Y_train.npy',Y_train)
np.save(folder + '/labels_train.npy',labels_train)

np.save(folder + '/rgb_valid.npy',X_valid)
np.save(folder + '/Y_valid.npy',Y_valid)
np.save(folder + '/labels_valid.npy',labels_valid)

np.save(folder + '/rgb_test.npy',X_test)
np.save(folder + '/Y_test.npy',Y_test)
np.save(folder + '/labels_test.npy',labels_test)

np.save(folder + '/X_views.npy',X_views)
np.save(folder + '/Y_multiview.npy',Y_multiview)
np.save(folder + '/labels_views.npy',labels_views)

