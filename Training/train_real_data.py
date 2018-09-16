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

from utils.save_test_res import *

img_size = 112
from models import Encoder_V, Encoder_S

# Method:
# 0 is to train from scratch
# 1 is to train from model pretrained ignoring symmetries
# 2 is to train from model pretrained considering symmetries
load = 0
considering_symmetries = True

numb_views = 80 #168
if numb_views == 80:
  from get_data.get_data_epson_80 import load_dataset
elif numb_views == 168:
  from get_data.get_data_epson_168 import load_dataset


def train(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model_S.train()
    model_V.train()
    tloss = 0
    analogic_loss = 0
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    iterations = 400 # 2000

    for index1b in range(iterations): 
      index1 = np.random.randint(len(X))
      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]

      symmetry = lab[0,2:5]

      num_views = len(imgs)

      batch = []
      sim = []
      cams = []
      var = np.random.randint(4)
      img1 = X[index1].reshape([img_size,img_size,3])
      for index2 in range( len(imgs) ):
        num_cam = int(lab[index2,1])
        
        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(img2)

        if considering_symmetry:
          # NOTE: There's a mismatch in the symmetries. For 80 views it's at position 4+ and for 168 it's at position 3+.
          # I'll try to fix it in the future.
          if numb_views == 80:
            sim.append(labels[index1,4 + num_cam - 1] )
          elif numb_views == 168:
            sim.append(labels[index1,3 + num_cam - 1] )
        else:
          similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2], [1,1,1]) # Calculate it again using symmetry 1
          sim.append(similarity)

        cams.append(num_cam)

      batch = np.array(batch)
      sim = np.array(sim)
      optims = np.sum( sim < np.min(sim) +0.1)
      cams = np.array(cams)
      order = np.argsort(cams)

      cams = cams[order]
      batch = batch[order]
      sim = sim[order]
      
      #sim = Find_dists_given_symmetry(sim, symmetry)
      #sim_norm = 0.15*sim/(np.max(sim) - np.min(sim))
      sim_norm = 0.2*sim/1.57
      #sim_norm = 0.2*sim/1.57/np.sum( sim < np.min(sim) + 0.01 ) 
    
      data_S = torch.from_numpy( np.transpose(img1.reshape([1,img_size,img_size,3]), [0,3,1,2]) ).float()
      data_V = torch.from_numpy( np.transpose(batch.reshape([-1,img_size,img_size,1]),[0,3,1,2]) ).float()
      target = torch.from_numpy( np.float32(sim_norm) )
      
      data_S, data_V, target = data_S.cuda(), data_V.cuda(), target.cuda()
      
      data_S, data_V, target = Variable(data_S), Variable(data_V), Variable(target)
      
      optimizer_S.zero_grad()
      optimizer_V.zero_grad()

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

    print('Train Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f}, Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, iterations, tloss/iterations, analogic_loss*1.0/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations ))

def test(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model_S.eval()
    model_V.eval()
    
    tloss = 0
    correct = 0
    analogic_loss = 0
    test_batch = 1
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    iterations =  len(X)

    for index1 in range( iterations ):
      
      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]

      batch = []
      sim = []
      cams = []
      num_views = len(imgs)

      var = np.random.randint(4)
      img1 = X[index1].reshape([img_size,img_size,3])
      for index2 in range( len(imgs) ):
        num_cam = int(lab[index2,1])
        #similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2], [1,1,1]) #symmetry)

        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(img2)
        if numb_views == 80:
          sim.append(labels[index1,4 + num_cam - 1] )
        elif numb_views == 168:
          sim.append(labels[index1,3 + num_cam - 1] )
        cams.append(num_cam)

      batch = np.array(batch)
      sim = np.array(sim)
      cams = np.array(cams)
      order = np.argsort(cams)
      
      cams = cams[order]
      batch = batch[order]
      sim = sim[order]

      #sim = Find_dists_given_symmetry(sim, symmetry)
      optims = np.sum( sim < np.min(sim) +0.1)
      sim_norm = 0.15*sim/(np.max(sim) - np.min(sim))
#      sim_norm = 0.2*sim/1.57

      data_S = torch.from_numpy( np.transpose(img1.reshape([1,img_size,img_size,3]), [0,3,1,2]) ).float()
      data_V = torch.from_numpy( np.transpose(batch.reshape([-1,img_size,img_size,1]),[0,3,1,2]) ).float()
      target = torch.from_numpy( np.float32(sim_norm) )
      
      data_S, data_V, target = data_S.cuda(), data_V.cuda(), target.cuda()
      
      data_S, data_V, target = Variable(data_S), Variable(data_V), Variable(target)
      
      optimizer_S.zero_grad()
      optimizer_V.zero_grad()

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
      
      #save_results(img1, batch, pred_order, order_sim, index1)
      
    print('Test Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, iterations, tloss/iterations, analogic_loss*1.0/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations ))
    return r5*1.0/iterations
    
if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  X_train,Y_train,labels_train,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views = load_dataset()  

  if load == 0:
    model_S = Encoder_S()
    model_V = Encoder_V()
  elif load == 1:
    model_S = torch.load('model_train_pretrained_nosymms_S_'+ str(numb_views))
    model_V = torch.load('model_train_pretrained_nosymms_V_'+ str(numb_views))
  elif load == 2:
    model_S = torch.load('model_train_pretrained_symms_S_'+ str(numb_views))
    model_V = torch.load('model_train_pretrained_symms_V_'+ str(numb_views))

  # Training settings
  model_S = model_S.cuda()
  model_V = model_V.cuda()

  optimizer_S = optim.Adam(model_S.parameters(), lr=2e-4, betas = (0.9, 0.999))
  optimizer_V = optim.Adam(model_V.parameters(), lr=2e-4, betas = (0.9, 0.999))

  best_acc = 0
  for epoch in range(1, epochs + 1):
      train(epoch,X_train,Y_train,labels_train,X_views,Y_multiview,labels_views)
      acc = test(epoch,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views)

      if acc > best_acc:
        best_epoch = epoch
        best_acc = acc
        print("")
        torch.save(model_S,'Model_type_' + str(load) + '_S_' + str(numb_views))
        torch.save(model_V,'Model_type_' + str(load) + '_V_' + str(numb_views))

      if epoch - best_epoch > 30:
        brek
