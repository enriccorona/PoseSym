from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from rotm2quat import quaternion_from_matrix
from eval_pair_symm import * #eval_pair
from transforms import random_transforms
#from get_data_epson_moreviews3 import load_test_dataset

#from get_data_epson_moreviews3_diffsplit import load_test_dataset # 168/648 views

from get_data_epson_80_time3_cropping import load_test_dataset # 80 views
#from get_data_epson_80_time3_cropping import load_valid_dataset as load_test_dataset # 80 views

from save_test_res import *

img_size = 112
from train_nonshared_deeper import Encoder_V, Encoder_S

def get_distribution_gt(epoch,X,Y,labels,X_views,Y_views,labels_views):
    
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
        #similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2], symmetry)

        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(img2)
        sim.append(labels[index1,4 + num_cam - 1] )
#        sim.append(labels[index1,3 + num_cam - 1] )
        cams.append(num_cam)

      batch = np.array(batch)
      sim = np.array(sim)
      cams = np.array(cams)
      order = np.argsort(cams)
      
      cams = cams[order]
      batch = batch[order]
      sim = sim[order]

      distances.append(np.min(sim))

    np.save('test_distances.npy',distances)
      
    print('Test Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Spheric distance: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f} - In 10 deg: {:.3f}, In 20 deg: {:.3f}, In 40 deg: {:.3f} '.format(
              epoch, iterations, tloss/iterations, analogic_loss*1.0/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations, minus10*1.0/iterations, minus20*1.0/iterations, minus40*1.0/iterations ))
    return r5*1.0/iterations
    #return acc*1.0/iterations
    
if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1
  X_test,Y_test,labels_test,X_views,Y_multiview,labels_views = load_test_dataset()  
  acc = get_distribution_gt(0,X_test,Y_test,labels_test,X_views,Y_multiview,labels_views)
  
