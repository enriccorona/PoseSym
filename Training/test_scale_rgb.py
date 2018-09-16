from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from utils.rotm2quat import quaternion_from_matrix
from utils.eval_pair_symm import * #eval_pair
from utils.transforms import random_transforms
from get_data.get_data_scale_3 import load_dataset
from utils.save_test_res import *


img_size = 112

#from train_symm_deeper import Encoder

#from train_nonshared_deeper2 import Encoder_V, Encoder_S

def train(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model_S.train()
    model_V.train()

    tloss = 0
    analogic_loss = 0
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    iterations = 1000 # 2000
    #data_save = []

    for index1 in range( iterations): 
      #ind = np.arange(len(X))
      #np.random.shuffle(ind)
      #X = X[ind]
      #Y = Y[ind]
      #labels = labels[ind]
      index1 = np.random.randint(len(X))

      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]

      if len(imgs) == 0:#!= 80:
        continue

      batch = []
      sim = []
      cams = []

      for index2 in range( len(imgs) ):
        num_cam = int(lab[index2,1])
        if num_cam in cams:
          continue

        if len(X[index1]) == 0:
          continue
        
        img1 = np.array(X[index1]).reshape([img_size,img_size,3])
        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append([img2])
        sim.append(labels[index1,3 + num_cam - 1] )
        cams.append(num_cam)
      
      batch = np.array(batch)
      sim = np.array(sim)
      optims = np.sum( sim < np.min(sim) +0.01)
      cams = np.array(cams)
      
      #sim = Find_dists_given_symmetry(sim, symmetry)
      #sim_norm = 0.15*sim/(np.max(sim) - np.min(sim))
      sim_norm = 0.2*sim/1.57
      order = np.argsort(sim)

      #data_save.append(np.float32( [ batch[0,0], batch[order[0],1], batch[order[1],1], batch[order[2],1], batch[order[3],1], batch[order[4],1], batch[order[5],1] ] ) )
      


      data_S = torch.from_numpy( np.transpose(img1.reshape([1,img_size,img_size,3]), [0,3,1,2]) ).float()
      data_V = torch.from_numpy( np.transpose(batch.reshape([-1,img_size,img_size,1]),[0,3,1,2]) ).float()
      target = torch.from_numpy( np.float32(sim_norm) )
      
      data_S, data_V, target = data_S.cuda(), data_V.cuda(), target.cuda()
      
      data_S, data_V, target = Variable(data_S), Variable(data_V), Variable(target)
      
      optimizer_S.zero_grad()
      optimizer_V.zero_grad()

      output_S = model_S(data_S)
      output_V = model_V(data_V)

      num_views = 80
      
      output_S = output_S.expand(num_views,64)
      
      output = torch.mul(output_V[:,:],output_S[:,:])
      output = torch.sum(output, 1)
      
      output = output.view( 1 , -1)
      
      output = F.softmax(output)
      
      _, pred_order = torch.sort(-1*output,1)
      
      order_sim = np.argsort(sim,0)
      #output = output.view( 1 , -1)
      
      #loss = F.nll_loss(output, Variable( torch.from_numpy( np.float32( [order_sim[0]] ) ).cuda().long() ) )
      loss = Variable( torch.FloatTensor(sim_norm - sim_norm[order_sim[0]]).cuda() )
      loss = torch.max( Variable( torch.zeros(len(sim)).cuda() ) , loss + output.view(-1) - output[0,order_sim[0]].repeat(num_views) )
      #loss = torch.max( Variable( torch.zeros(len(sim)).cuda() ) , alpha + output.view(-1) - output[0,order_sim[0]].repeat(num_views) )
      #loss = torch.mul( Variable( torch.FloatTensor(sim - sim[order_sim[0]]).cuda() ), loss)
      loss = loss.sum()

      # Compute gradients and train
      loss.backward()
      optimizer_S.step()
      optimizer_V.step()

      analogic_loss += sim[pred_order.cpu().data.numpy()[0,0]]
      res = pred_order.cpu().data.numpy()

      acc += 1*(res[0,0] in order_sim[0:optims])
      r3 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:3] ))
      r5 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:5] ))
      r10 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:10]  ))

      tloss += loss.cpu().data.numpy()[0]

      #if save == True and index1b < 20:
      #  data_rgb.append( img1 )

    #np.save('data.npy',data_save)
    print('Train Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f}, Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, iterations, tloss/iterations, analogic_loss*1.0/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations ))

def test(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model_S.eval()
    model_V.eval()

    all_list = []
    all_sims = []
    num_views = 80

    tloss = 0
    correct = 0
    analogic_loss = 0
    test_batch = 1
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0

    for index1 in range( len(X) ):
      
      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]
      #symmetry = lab[0,2:5]

      batch = []
      sim = []
      cams = []

      #print(len(imgs))
      if len(imgs) == 0:#!= 80:
        continue

      var = np.random.randint(4)
      img1 = X[index1].reshape([img_size,img_size,3])

      for index2 in range( len(imgs) ):
        num_cam = int(lab[index2,1])
        if num_cam in cams:
          continue

        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(img2)
        sim.append(labels[index1,3+num_cam-1])
        cams.append(num_cam)

      batch = np.array(batch)
      sim = np.array(sim)
      optims = np.sum( sim < np.min(sim) +0.1)
      cams = np.array(cams)
      
      #sim = Find_dists_given_symmetry(sim, symmetry)
      #sim_norm = 0.15*sim/(np.max(sim) - np.min(sim))
      sim_norm = 0.2*sim/1.57

      data_S = torch.from_numpy( np.transpose(img1.reshape([1,img_size,img_size,3]), [0,3,1,2]) ).float()
      data_V = torch.from_numpy( np.transpose(batch.reshape([-1,img_size,img_size,1]),[0,3,1,2]) ).float()
      target = torch.from_numpy( np.float32(sim_norm) )
      
      data_S, data_V, target = data_S.cuda(), data_V.cuda(), target.cuda()
      
      data_S, data_V, target = Variable(data_S), Variable(data_V), Variable(target)
      
      output_S = model_S(data_S)
      output_V = model_V(data_V)

      output_S = output_S.expand(num_views,64)

      output = torch.mul(output_V[:,:],output_S[:,:])
      output = torch.sum(output, 1)
      
      output = output.view( 1 , -1)
      
      output = F.softmax(output)
      
      _, pred_order = torch.sort(-1*output,1)
      
      order_sim = np.argsort(sim,0)
      #output = output.view( 1 , -1)
      
      #loss = F.nll_loss(output, Variable( torch.from_numpy( np.float32( [order_sim[0]] ) ).cuda().long() ) )
      loss = torch.max( Variable( torch.zeros(len(sim)).cuda() ) , alpha + output.view(-1) - output[0,order_sim[0]].repeat(num_views) )
      loss = torch.mul( Variable( torch.FloatTensor(sim - sim[order_sim[0]]).cuda() ), loss)
      loss = loss.sum()

      analogic_loss += sim[pred_order.cpu().data.numpy()[0,0]]
      res = pred_order.cpu().data.numpy()

      acc += 1*(res[0,0] in order_sim[0:optims])
      r3 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:3] ))
      r5 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:5] ))
      r10 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:10]  ))
      
      tloss += loss.cpu().data.numpy()[0]
      
      #save_results(img1, batch, pred_order, order_sim[:optims], index1, sim[pred_order.cpu().data.numpy()[0,0:4]], batch[order_sim[0]])
      all_list.append( np.min(sim[pred_order.cpu().data.numpy()[0,0:4]] ) )
      all_sims.append(sim[pred_order.cpu().data.numpy()[0,0:4]])
      
    np.save('test_res3/sims.npy',all_sims)
    np.save('test_res3/order.npy',all_list)

    print('Test Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, len(X), tloss/len(X), analogic_loss*1.0/len(X), acc*1.0/len(X), r3*1.0/len(X), r5*1.0/len(X), r10*1.0/len(X) ))
    #return acc*1.0/len(X)
    return r5*1.0/len(X)
    
if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  print("Loading data")
  #X_scene,Y_scene,rgb_scene,labels_scene,X_views,Y_multiview,labels_views = load_dataset()
  _,Y_train,X_train,labels_train,_,Y_test,X_test,labels_test,X_views,Y_multiview,labels_views = load_dataset()  
  #X_train,Y_train,labels_train,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views = load_dataset()  

  #numb = int(len(X_scene)*0.7)
  #numb_test = len(X_scene)

  #X_train = X_scene[:numb]
  #Y_train = Y_scene[:numb]
  #labels_train = labels_scene[:numb]

  #X_test = X_scene[numb:numb_test]
  #Y_test = Y_scene[numb:numb_test]
  #labels_test = labels_scene[numb:numb_test]
  #rgb_test = rgb_scene[numb:numb_test]
  
  #X_scene = None
  #rgb_scene = None
  
  # Training settings
  model_S = torch.load('model_train_scalewithsymms_S_Nov2') #'model_train_nonshared_symm_more_S') #Encoder_S()
  model_V = torch.load('model_train_scalewithsymms_V_Nov2') #'model_train_nonshared_symm_more_v') #model_train_nonshared_V') #Encoder_V()

  model_S = model_S.cuda()
  model_V = model_V.cuda()

  optimizer_S = optim.Adam(model_S.parameters(), lr=2e-4, betas = (0.9, 0.999))
  otimizer_V = optim.Adam(model_V.parameters(), lr=2e-4, betas = (0.9, 0.999))
  #optimizer_S = optim.Adam(model_S.parameters(), lr=1e-5, betas = (0.9, 0.999))
  #optimizer_V = optim.Adam(model_V.parameters(), lr=1e-5, betas = (0.9, 0.999))
  print(np.shape(X_train))
  best_acc = 0
  acc = test(1,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views)
  

