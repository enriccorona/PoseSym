from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from IPython import embed
import cv2
from utils.rotm2quat import quaternion_from_matrix
from utils.eval_pair_symm import * #eval_pair
from utils.transforms import random_transforms

img_size = 112

from train_to_predict_symm_lossinsymmetry import get_probability_order

from get_data.get_data_scale_3 import load_dataset
from train_to_predict_sym_and_pose import MLP
from get_data.get_data_more_views_epson import load_epson_views
from get_data.get_data_more_views import load_views

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

numb_views = 80
if numb_views == 80:
  equivals = np.load('pairs.npy') # 80 views
  x_,x_2,x_4,y_,y_2,y_4,z_,z_2,z_4 = equivals
elif numb_views == 168:
  equivals = np.load('pairs_168.npy') # 168 views
  x,x_2,x_3,x_4,y,y_2,y_3,y_4,z,z_2,z_3,z_4 = equivals

x_ = [k for k in x if k not in x_4 ]
y_ = [k for k in y if k not in y_4 ]
z_ = [k for k in z if k not in z_4 ]

x_4 = [k for k in x_4 if k not in x_2 ]
y_4 = [k for k in y_4 if k not in y_2 ]
z_4 = [k for k in z_4 if k not in z_2 ]

from save_test_res import *

def predict_and_save(X,Y,labels, test = False):
    #model_V.train()
   
    debug = False
    tloss = 0
    correct = 0
    analogic_loss = 0
    test_batch = 1
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    recall = 0
    precision = 0
    all_f1 = 0

    recall_num = 0
    recall_den = 0
    precision_num = 0
    precision_den = 0

    results = []
    sims = []
    models = np.unique(labels[:,0])
    saved = 0

    iterations = len(models)
    
    symmetries_inferred = []
    meshes_done = []
  
    import time
    for index1 in range( iterations):

      mesh = models[index1]
      starting_time = time.time()

      print(mesh)
# RETRIEVE IN DATABASE VIEWS CONTAINING MESH
      nums_mesh = (labels[:,0] == mesh)
      imgs = X[nums_mesh]
      lab = labels[nums_mesh]
      num_views = len(imgs)
    
# PREPARE NETWORK INPUTS
      batch = []
      cams = []
# NOT OPTIMAL WAY, JUST TAKING ADVANTAGE OF CODE DONE FOR COMPUTING SCENE DISTANCE TO VIEW AT EACH STEP
      #for index2 in range( jjj,jjj+1 ):#num_views ):
      for index2 in range( num_views ):
        num_cam = int(lab[index2,1])
        batch.append(imgs[index2].reshape([1,1,112,112]))
        cams.append(num_cam)

      cams = np.array(cams)
      batch = np.float32(batch)

      order = np.argsort(cams)
      cams = cams[order]
      batch = batch[order] 

# OBTAIN GROUND TRUTH
      alpha = 0.0125 #Y.mean()

# CONVERT GROUND TRUTH + VIEWS TO TENSORS
      data_V = torch.from_numpy(batch)
      #data_V = torch.from_numpy(input_net)
      #data_V = torch.from_numpy(batch.reshape([-1,1,img_size,img_size])).float()
      data_V = data_V.cuda()
      data_V = Variable(data_V)

      output_V = model_V(data_V)
      #k = torch.mul(output_V,output_V)
      #k = torch.sum(k,1)

      #k = torch.sum( output_V*output_V, 1 )
#      output_V = output_V/(torch.sqrt(k.view([-1,1])).repeat(1,64) ) 

      #k = torch.sum( output_V*output_V, 1 )
      #print(np.mean(k.cpu().data.numpy()))

      #if np.mean(k.cpu().data.numpy()) != np.mean(k.cpu().data.numpy()):
      #  afhau

      maximum_zero = Variable( torch.zeros(1).float().cuda() ) + 0.001
      probs_ = Variable( torch.zeros([80,80]).cuda() )
      all_combinations = []  
      for index2 in range(80):
        num_cam2 = int(lab[index2,1])
        this_combination = []
        for index3 in range(80):
          num_cam3 = int(lab[index3,1])
#TODO: CHECK IF SHOULD BE num_cam3 IN THE FIRST OR IN THE SECOND
          val = output_V.cpu().data.numpy()
          similarity = torch.mul(output_V[num_cam3-1,:], output_V[num_cam2-1,:])
          similarity = torch.sum(similarity)

          #similarity = similarity/2 + 0.49
#          similarity = similarity/2 + 0.5

          probs_[num_cam2-1, num_cam3-1] = similarity

      probs_ = torch.sigmoid(BN( probs_.view([-1,1]) ))
      
      all_combinations = probs_.view([80,80]) #BN(all_combinations)


      val_x2 = get_probability_order(all_combinations,x_2)
      val_x4 = get_probability_order(all_combinations,x_4)
      val_x = get_probability_order(all_combinations,x_)

      val_y2 = get_probability_order(all_combinations,y_2)
      val_y4 = get_probability_order(all_combinations,y_4)
      val_y = get_probability_order(all_combinations,y_)

      val_z2 = get_probability_order(all_combinations,z_2)
      val_z4 = get_probability_order(all_combinations,z_4)
      val_z = get_probability_order(all_combinations,z_)
      
      x = torch.stack([val_x2,val_x,val_x])
      y = torch.stack([val_y2,val_y4,val_y])
      z = torch.stack([val_z2,val_z,val_z])
      symmetry_inferred = torch.stack([x,y,z]).view([3,3])

      order_x,order_y,order_z = model_mlp(x,y,z,test)
      
      order_x = order_x.cpu().data.numpy()[0]
      order_y = order_y.cpu().data.numpy()[0]
      order_z = order_z.cpu().data.numpy()[0]

      new_symmetry = infer_symmetry_energy(order_x,order_y,order_z)

      considered = [1,2,4,np.Inf]
      new_symmetry = [considered[new_symmetry[0]], considered[new_symmetry[1]], considered[new_symmetry[2]]]

      print(new_symmetry)
      meshes_done.append(mesh)
      symmetries_inferred.append(new_symmetry)

      final_time = time.time()
      print("It took: ")
      print( final_time - starting_time )

    np.save('meshes_inferred_Nov_last.npy',meshes_done)
    np.save('symmetries_inferred_Nov_last.npy', symmetries_inferred)

    return 1 #(1 - acc/iterations)


if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1
  model_V = torch.load('model_predict_symmetries_V_' + str(numb_views))
  BN = torch.load('model_predict_symmetries_BN_' + str(numb_views))
  model_mlp = torch.load('model_predict_symmetries_mlp_' + str(numb_views))

  _,_,_,_,_,_,_,_,X_views,Y_multiview,labels_views = load_dataset()
  model_mlp = model_mlp.cuda()
  BN = BN.cuda()
  model_V = model_V.cuda()

  print(X_views.max())
  best_acc = 0
  acc = predict_and_save(X_views,Y_multiview,labels_views,test = True)


