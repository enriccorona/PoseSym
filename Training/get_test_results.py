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

from models import Encoder_V, Encoder_S

from utils.save_test_res import *

numb_views = 80 #168
if numb_views == 80:
  from get_data.get_data_epson_80 import load_test_dataset
elif numb_views == 168:
  from get_data.get_data_epson_168 import load_test_dataset
loadmodel_V = ''
loadmodel_S = ''
top = 4
img_size = 112

def test(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model_S.eval()
    model_V.eval()
    all_sims = []
    
    all_list = []
    tloss = 0
    correct = 0
    analogic_loss = 0
    test_batch = 1
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    minus10 = 0
    minus20 = 0
    minus40 = 0
    distances = []
    iterations = len(X) #min(500,len(X))

    for index1 in range( iterations ):
      
      mesh = labels[index1,0]
      #if mesh == 16:
        #continue
      cam = labels[index1,2]
      print(mesh)

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
        #similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2], symmetry)

        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(img2)
        sim.append(labels[index1,4 + num_cam - 1] )
        #sim.append(labels[index1,3 + num_cam - 1] )
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
      #print("")
      #print(mesh)
      #print(optims)
#      sim_norm = 0.15*sim/(np.max(sim) - np.min(sim))
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

      analogic_loss += np.min(sim[pred_order.cpu().data.numpy()[0,0:top]])
      res = pred_order.cpu().data.numpy()

      acc += 1*(res[0,0] in order_sim[0:optims])
      r3 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:3] ))
      r5 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:5] ))
      r10 += 1*any((True for x in order_sim[0:optims] if x in res[0,0:10]  ))
      
      print(np.min(sim[pred_order.cpu().data.numpy()[0,0:top]]))
      minus10 += 1*( np.min(sim[pred_order.cpu().data.numpy()[0,0:top]])*180.0/math.pi < 10.0 )
      minus20 += 1*( np.min(sim[pred_order.cpu().data.numpy()[0,0:top]])*180.0/math.pi < 20.0 )
      minus40 += 1*( np.min(sim[pred_order.cpu().data.numpy()[0,0:top]])*180.0/math.pi < 40.0 )
      
      tloss += loss.cpu().data.numpy()[0]
      save_results(img1, batch, pred_order, order_sim[:optims], index1)
      #save_results(img1, batch, pred_order, order_sim[:optims], index1, sim[pred_order.cpu().data.numpy()[0,0:4]] )
      #save_results(img1, batch, pred_order, order_sim[:optims], index1, sim[pred_order.cpu().data.numpy()[0,0:4]], batch[order_sim[0]] )
      distances.append(np.min(sim[pred_order.cpu().data.numpy()[0,0:top]]))

      all_list.append( np.min(sim[pred_order.cpu().data.numpy()[0,0:4]] ) )
      all_sims.append(sim[pred_order.cpu().data.numpy()[0,0:4]])
      
    np.save('test_res3/sims.npy',all_sims)
      
    np.save('test_res/order.npy',all_list)



    np.save('test_distances.npy',distances)
      
    print('Test Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f} - In 10 deg: {:.3f}, In 20 deg: {:.3f}, In 40 deg: {:.3f} '.format(
              epoch, iterations, tloss/iterations, analogic_loss*1.0/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations, minus10*1.0/iterations, minus20*1.0/iterations, minus40*1.0/iterations ))
    return r5*1.0/iterations
    #return acc*1.0/iterations
    
if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  X_test,Y_test,labels_test,X_views,Y_multiview,labels_views = load_test_dataset()  

  # Training settings


  
  model_S = torch.load(loadmodel_S)
  model_V = torch.load(loadmodel_V)

  model_S = model_S.cuda()
  model_V = model_V.cuda()
  
  acc = test(0,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views)
  
