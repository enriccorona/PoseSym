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

import time

from utils.utils_trainsymmetry import get_probability_order

from train_to_predict_sym_and_pose import MLP
from get_data.get_data_more_views_epson import load_epson_views
from get_data.get_data_more_views import load_views

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TRAINING FOR:
numb_views = 80
# CONSIDERING THEORETICAL LIMITS:
theory = False
img_size = 112

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

def train(X,Y,labels,meshes,symms, test = False):
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

    recall_den_0 = 0
    precision_num_0 = 0
    precision_den_0 = 0

    recall_den_1 = 0
    precision_num_1 = 0
    precision_den_1 = 0

    recall_den_2 = 0
    precision_num_2 = 0
    precision_den_2 = 0

    recall_den_3 = 0
    precision_num_3 = 0
    precision_den_3 = 0

    recall_den_4 = 0
    precision_num_4 = 0
    precision_den_4 = 0


    recall_num = 0
    recall_den = 0
    precision_num = 0
    precision_den = 0

    results = []
    sims = []
    models = np.unique(meshes)
    saved = 0

# GET WEIGHTS
    weights = np.array([0,0,0,0])
    for index1 in range(len(models)):
      mesh = models[index1]
      ind = np.argmax(mesh == meshes)
      symmetry = symms[ind]
      Vsymmetry = np.argmax(symmetry.reshape([-1,1]) == np.array([0,2,4,np.Inf]*3).reshape([3,4]),1)
      weights[Vsymmetry] += 1

    total = 1.0*weights.sum()
    #weights = 1 - weights*1.0/total
    weights[weights==0] = total
    weights = total/weights
    print(weights)
    weights = torch.from_numpy( weights ).float().cuda() 

    if test:
      iterations = len(models)
    else:
      iterations = 20
    #print(iterations)
    ##print(models)
    
    for num_iteration in range( iterations):

      if test:
        index1 = num_iteration
      else:
        index1 = np.random.randint(len(models))

      mesh = models[index1]

# RETRIEVE IN DATABASE VIEWS CONTAINING MESH
      nums_mesh = (labels[:,0] == mesh)
      imgs = X[nums_mesh]
      lab = labels[nums_mesh]
      num_views = len(imgs)
    
      ind = np.argmax(mesh == meshes)
      symmetry = symms[ind]

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
      Y = find_equivalences_from_symm(symmetry)
      alpha = 0.0125 #Y.mean()

# CONVERT GROUND TRUTH + VIEWS TO TENSORS
      data_V = torch.from_numpy(batch)
      #data_V = torch.from_numpy(input_net)
      #data_V = torch.from_numpy(batch.reshape([-1,1,img_size,img_size])).float()
      Y = torch.from_numpy( Y ) #.float()
      #data_V = torch.from_numpy( np.transpose(batch.reshape([-1,img_size,img_size,1]),[0,3,1,2]) ).float()
      #data_V = torch.from_numpy( batch.reshape([-1,1,img_size,img_size] ) ).float()
      Y, data_V = Y.cuda(), data_V.cuda()
      Y, data_V = Variable(Y), Variable(data_V)

      optimizer_V.zero_grad()
      optimizer_BN.zero_grad()
      optimizer_V.zero_grad()

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
          

         # similarity = torch.sigmoid( (similarity - 0.99)/0.002 )
         # this_combination.append(torch.max( maximum_zero ,similarity ))
          #this_combination.append( similarity )

      # STACK VECTOR TO OBTAIN TENSOR AND APPEND
        #all_combinations.append( torch.stack(this_combination) )

    # STACK TO OBTAIN TENSOR OF ALL COMBINATIONS (80x80)
      #all_combinations = torch.stack(all_combinations,0)
      #all_combinations = torch.stack(all_combinations).view([-1,1])

      
 #     print(all_combinations.data.size())
 #     print(all_combinations)
 #     print(BN)
      
      #maximum_zeros = Variable((torch.ones([80,80])).cuda())
      #maximum_zeros = maximum_zeros*maximum_zero
      #embed()
      #probs_ = hard(BN( probs_.view([-1,1]) ) )
      probs_ = torch.sigmoid(BN( probs_.view([-1,1]) ))
      #probs_ = torch.sigmoid(BN( probs_.view([-1,1]) ))
      #probs_ = BN( probs_.view([-1,1]) )
      #probs_ = torch.clamp(probs_, max=0.9,min=0.1)
      #probs_ = torch.sigmoid(BN( probs_.view([-1,1]) ))
#      probs_ = (probs_ - 0.5)*0.9 + 0.5


      
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
      #print(symmetry)
      #print(val_x2)
      #print(val_x4)
      #print(val_x)
      
      x = torch.stack([val_x2,val_x,val_x])
      y = torch.stack([val_y2,val_y4,val_y])
      z = torch.stack([val_z2,val_z,val_z])
      symmetry_inferred = torch.stack([x,y,z]).view([3,3])



      

      #embed()
      #symm_gt = 1*np.array([[symmetry[0] == 2 or symmetry[0] >= 4,symmetry[0] >= 4,symmetry[0] > 4 ],[symmetry[1]==2 or symmetry[1] >= 4,symmetry[1]>=4,symmetry[1]>4],[symmetry[2]==2 or symmetry[2]>=4,symmetry[2]>=4,symmetry[2]>4]])
      #print(symm_gt)
      #symm_gt = Variable( torch.from_numpy(symm_gt).float().cuda())
      
      #loss = - symm_gt*torch.log(symmetry_inferred) - (1-symm_gt)*torch.log(1-symmetry_inferred)
      #loss = loss.mean()*3


# LOSS MLP:
      
      order_x,order_y,order_z = model_mlp(x,y,z,test)
      

#      CEx = -1.0*(classes_symmetry_x==0)*torch.log(order_x[0,0]) - 1.0*(classes_symmetry_x == 1)*torch.log(order_x[0,1]) - 1.0*(classes_symmetry_x == 2)*torch.log(order_x[0,2]) - 1.0*(classes_symmetry_x == 3)*torch.log(order_x[0,3])
#      CEy = - 1.0*(classes_symmetry_y==0)*torch.log(order_y[0,0]) - 1.0*(classes_symmetry_y == 1)*torch.log(order_y[0,1]) - 1.0*(classes_symmetry_y == 2)*torch.log(order_y[0,2]) - 1.0*(classes_symmetry_y == 3)*torch.log(order_y[0,3])
#      CEz = - 1.0*(classes_symmetry_z==0)*torch.log(order_z[0,0]) - 1.0*(classes_symmetry_z == 1)*torch.log(order_z[0,1]) - 1.0*(classes_symmetry_z == 2)*torch.log(order_z[0,2]) - 1.0*(classes_symmetry_z == 3)*torch.log(order_z[0,3])
      
      symmetry[symmetry == 0] = 1
      symmetryprime = symmetry[:]
      symmetryprime[symmetryprime > 4] = np.Inf
      Vsymmetry = np.argmax(symmetry.reshape([-1,1]) == np.array([1,2,3,4,np.Inf]*3).reshape([3,5]),1)
      

      Vsymmetry = np.argmax(symmetry.reshape([-1,1]) == np.array([1,2,4,np.Inf]*3).reshape([3,4]),1)
      #Y = Y.long()
      #print(Y)
      
#      BCE = torch.nn.CrossEntropyLoss(weights)
#      Vsymmetry = Variable(torch.from_numpy(Vsymmetry)).long().cuda()
      #Vsymmetry = Vsymmetry.long()

      #embed()
      #loss_x = BCE(order_x,Vsymmetry[0].view([-1]))
      #loss_y = BCE(order_y,Vsymmetry[1].view([-1]))
      #loss_z = BCE(order_z,Vsymmetry[2].view([-1]))

      #embed()
      loss_x = - 1*(Vsymmetry[0] == 0)*torch.log(order_x[0,0])*weights[0] - 1*(Vsymmetry[0] == 1)*torch.log(order_x[0,1])*weights[1] - 1*(Vsymmetry[0] == 2)*torch.log(order_x[0,2])*weights[2] - 1*(Vsymmetry[0] == 3)*torch.log(order_x[0,3])*weights[3]
      loss_y = - 1*(Vsymmetry[1] == 0)*torch.log(order_y[0,0])*weights[0] - 1*(Vsymmetry[1] == 1)*torch.log(order_y[0,1])*weights[1] - 1*(Vsymmetry[1] == 2)*torch.log(order_y[0,2])*weights[2] - 1*(Vsymmetry[1] == 3)*torch.log(order_y[0,3])*weights[3]
      loss_z = - 1*(Vsymmetry[2] == 0)*torch.log(order_z[0,0])*weights[0] - 1*(Vsymmetry[2] == 1)*torch.log(order_z[0,1])*weights[1] - 1*(Vsymmetry[2] == 2)*torch.log(order_z[0,2])*weights[2] - 1*(Vsymmetry[2] == 3)*torch.log(order_z[0,3])*weights[3]
#      print(weights)

      #print(Vsymmetry)
      #Val_1Y[ torch.lt(Val_1Y,-9999)] #- 1.0*Variable(torch.ones(1).cuda())*9999
      #print("Val_1Y")

      loss_mlp = loss_x.mean() + loss_y.mean() + loss_z.mean()
#      loss += loss_mlp 
      loss = loss_mlp

      #loss_mlp = CEx.mean() + CEy.mean() + CEz.mean()
  #    print(loss)
 #     print(loss_mlp)
  #    print("")




      tloss += loss.cpu().data.numpy()[0]

      a = (Y.cpu().data.numpy() > 0.5)
      b = (all_combinations.cpu().data.numpy() > 0.5) 

      acc += np.logical_xor( a, b).mean()
      precision += 1.0*np.logical_and( a, b).sum()/np.sum( b + 0.001 )
      recall += 1.0*np.logical_and( a, b).sum()/np.sum( a )
      #print("xor")
      #print( np.logical_xor( a, b) )
      #print("")


      #print("Retrieved documents")
      #print(np.sum( all_combinations.cpu().data.numpy() > 0.5 ))
      #print("Relevant documents")
      #print(np.sum( Y.cpu().data.numpy() > 0.5 ))
      
      #if test and np.max(symmetry) > 1:
      #  print(symmetry)
      #  print("Y")
      #  print(Y.cpu().data.numpy().tolist())
      #  print("Pred")
      #  print(all_combinations.cpu().data.numpy().tolist())
      
      #if test:
      if True:
        #new_symmetry = infer_symmetry(all_combinations.cpu().data.numpy(),False)
        #new_symmetry = 
        order_x = order_x.cpu().data.numpy()[0]
        order_y = order_y.cpu().data.numpy()[0]
        order_z = order_z.cpu().data.numpy()[0]

      #  print(mesh)
      #  print(order_x)
      #  print(order_y)
      #  print(order_z)
      #  print("GT:")
      #  print(symmetry)

        considered = [1,2,4,np.Inf]

        # NOT CONSIDERING THEORY
        if not theory:
          new_symmetry = np.array([considered[np.argmax(order_x)], considered[np.argmax(order_y)], considered[np.argmax(order_z)]])
        # CONSIDERING THEORETICAL LIMITS:
        else:
          new_symmetry = infer_symmetry_energy(order_x,order_y,order_z)
          new_symmetry = [considered[new_symmetry[0]], considered[new_symmetry[1]], considered[new_symmetry[2]]]

          new_symmetry = np.array([considered[np.argmax(order_x.cpu().data.numpy())], considered[np.argmax(order_y.cpu().data.numpy())], considered[np.argmax(order_z.cpu().data.numpy())]])

        new_symmetry = np.array(new_symmetry)
        new_symmetry[new_symmetry>10] = 10
        symmetry[symmetry>10] = 10
        new_symmetry = np.int32(new_symmetry)
        symmetry = np.int32(symmetry)


        precision_num_0 += np.sum( np.logical_and(symmetry == 1, new_symmetry == 1)*1 )
        precision_num_1 += np.sum( np.logical_and(symmetry == 2, new_symmetry == 2)*1 )
        precision_num_2 += np.sum( np.logical_and(symmetry == 3, new_symmetry == 3)*1 )
        precision_num_3 += np.sum( np.logical_and(symmetry == 4, new_symmetry == 4)*1 )
        precision_num_4 += np.sum( np.logical_and(symmetry == 10, new_symmetry == 10)*1 )
        precision_den_0 += np.sum(new_symmetry==1)
        precision_den_1 += np.sum(new_symmetry==2)
        precision_den_2 += np.sum(new_symmetry==3)
        precision_den_3 += np.sum(new_symmetry==4)
        precision_den_4 += np.sum(new_symmetry==10)
        recall_den_0 += np.sum(symmetry==1)        
        recall_den_1 += np.sum(symmetry==2)        
        recall_den_2 += np.sum(symmetry==3)        
        recall_den_3 += np.sum(symmetry==4) 
        recall_den_4 += np.sum(symmetry==10)        

      if not test:
        loss.backward()
        optimizer_V.step()
        optimizer_BN.step()
        optimizer_mlp.step()

    recall_symmetries = ((precision_num_0+0.0001)/(0.0001+recall_den_0) + (precision_num_1+0.0001)/(0.0001+recall_den_1) + (precision_num_2+0.0001)/(0.0001+recall_den_2) + (precision_num_3+0.0001)/(0.0001+recall_den_3) + (precision_num_4+0.0001)/(0.0001+recall_den_4))/5
    precision_symmetries = ((precision_num_0+0.0001)/(0.0001+precision_den_0) + (precision_num_1+0.0001)/(0.0001+precision_den_1) + (precision_num_2+0.0001)/(0.0001+precision_den_2) + (precision_num_3+0.0001)/(0.0001+precision_den_3) + (precision_num_4+0.0001)/(0.0001+precision_den_4))/5

    F1_symmetries = 2.0*precision_symmetries*recall_symmetries/(precision_symmetries + recall_symmetries + 0.00001)
    
    
    recall = recall*1.0/iterations
    precision = precision*1.0/iterations
    
    print("LOSS: " + str(tloss/iterations) + " Accuracy " + str(1 - acc/iterations) + ", Recall " + str(recall) + ", Precision " + str(precision) + ", F1 " + str(2*precision*recall/(precision+recall + 0.00001)) + " | Recall " + str(recall_symmetries) + ", Precision " + " = " + str(precision_symmetries) + ", F1 " + str(F1_symmetries) )
    
    return F1_symmetries #(1 - acc/iterations)

def save_negative_sample(Y,gt,batch):
  mistakes =  np.logical_xor( Y > 0.5, gt > 0.5)
  mistakes_saved = 0
  while(mistakes_saved < 10):
    if mistakes.sum() == 0:
      return
    mistakes_row = mistakes.reshape(-1)
    index = np.argmax(mistakes_row)
    index_x = index/80
    index_y = index%80

    mistakes[index_y,index_x] = 0
    mistakes[index_x,index_y] = 0

    img1 = batch[index_x].reshape([112,112])
    img2 = batch[index_y].reshape([112,112])

    mistakes_saved += 1

    plt.close()
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1,2,1)
    plt.imshow( img1 )
    plt.axis('off')
    
    plt.subplot(1,2,2)
    
    plt.imshow( img2 )
    plt.axis('off')
    plt.title(str(gt[index_y,index_x]))

    plt.savefig('test_res/pair_'+str(mistakes_saved)+'.png')


if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  print("Loading data")
  X_views,Y_multiview,labels_views = load_views()
  X_views_epson,Y_multiview_epson,labels_views_epson = load_epson_views()

  print("Joining")
  from utils_trainsymmetry import join_splits
  X,Y,labels, meshes_train,meshes_valid,meshes_test, symms_train,symms_valid,symms_test = join_splits(X_views,Y_multiview,labels_views, X_views_epson,Y_multiview_epson,labels_views_epson)
  
  # Training settings
  model_V = torch.load('model_train_scalewithoutsymms_V')
  BN = nn.Linear(1,1) # NOTE Or was Batch Normalization??

# Initialize MLP to logical values:
  model_mlp = MLP()
  model_mlp.fc_x.bias.data = torch.from_numpy(np.array([6.0,-5.0,-5.0,-5.0])).float()
  model_mlp.fc_x.weight.data = torch.from_numpy(np.array( [[-5.0,-5.0,-5.0],[10.0,-1.0,-1.0],[0.0,10.0,-5.0],[0.0,0.0,10.0]])).float()

  model_mlp = model_mlp.cuda()
  BN = BN.cuda()
  model_V = model_V.cuda()

  optimizer_V = optim.Adam(model_V.parameters(), lr=2e-4, betas = (0.9, 0.999))
  optimizer_mlp = optim.Adam(model_mlp.parameters(),lr = 2e-3, betas = (0.9,0.999))
  optimizer_BN = optim.Adam(BN.parameters(), lr=2e-4, betas = (0.9, 0.999))

  starting_time = time.time()
  best_acc = 0
  for epoch in range(1, epochs + 1):
    # TRAIN ON SYNTHETIC OBJS
      train(X,Y,labels,meshes_train,symms_train)
    # TEST ON EPSON OBJS
      acc = train(X,Y,labels,meshes_valid,symms_valid,test = True)

      if acc >= best_acc:
        best_acc = acc
        print("BEST RESULTS IN VALIDATION. FOR TEST:")
        acc = train(X,Y,labels,meshes_test,symms_test,test = True)
        torch.save(model_V, 'model_predict_symmetries_V_' + str(numb_views))
        torch.save(BN, 'model_predict_symmetries_BN_' + str(numb_views))
        torch.save(model_mlp, 'model_predict_symmetries_mlp_' + str(numb_views))

      print(time.time() - starting_time)
      print("\n\n") 
