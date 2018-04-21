from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import cv2
from rotm2quat import quaternion_from_matrix
from eval_pair_symm import * #eval_pair
from transforms import random_transforms

def get_probability_order(predictions,pairs):

  if len(pairs) == 0:
    return Variable(torch.zeros(1).float().cuda())

  indexs1 = np.array(pairs)[:,0]-1
  indexs2 = np.array(pairs)[:,1]-1

  prob = Variable(torch.zeros(1).float().cuda())
  
  for j in range(len(indexs1)):
    prob += predictions[indexs1[j],indexs2[j]]

  prob = prob/len(indexs1)

  return prob
  
def join_splits(X_views,Y_multiview,labels_views, X_views_epson,Y_multiview_epson,labels_views_epson):
    list_meshes = np.array([134,2922,135,5481,223,224,2960] + [4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012,4014,4015,4016,4017,4018,4019] + [5009,5010,5011] )
    list_symms = np.array([[1,1,1],[2,2,2],[1,1,1],[1,np.Inf,1],[1,1,1],[1,1,1],[1,np.Inf,1]] + [[2,2,np.Inf]]*18 + [[1,1,np.Inf]]*3 )
    from correspondance_epson_models import get_correspondances#_strict
    list_meshes_epson, list_symms_epson, folders = get_correspondances() #_strict()

    X = np.concatenate((X_views,X_views_epson))
    Y = np.concatenate((Y_multiview,Y_multiview_epson))
    labels = np.concatenate((labels_views,labels_views_epson[:,0:2]))  
    symms = np.concatenate((list_symms,list_symms_epson))
    meshes = np.concatenate((list_meshes,np.arange(len(list_meshes_epson))))

    ind = [22, 33, 43,  7, 28, 30, 13, 16, 17, 37, 10, 14, 34,  1, 19, 44,  0, 23, 36, 27, 20, 35, 18, 42,  2,  8, 39, 26, 21, 32, 29,  3, 15,  4,  5, 25, 40, 38, 11, 41, 12, 31,  9,  6, 24]

    meshes = meshes[ind]
    symms = symms[ind]

    train_ = 25
    valid_ = 35

    meshes_train = meshes[:train_]
    meshes_valid = meshes[train_:valid_]
    meshes_test = meshes[valid_:]

    symms_train = symms[:train_]
    symms_valid = symms[train_:valid_]
    symms_test = symms[valid_:]

    return X,Y,labels, meshes_train,meshes_valid,meshes_test, symms_train,symms_valid,symms_test
