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
from eval_pair_symm import eval_pair
#from eval_pair import eval_pair
from transforms import random_transforms
#from get_data_depth_conserving import load_dataset
from get_data_more_views_epson import load_views as load_dataset

img_size = 112

class Encoder(nn.Module):
    def __init__(self):
        import nninit
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(576, 64)
        self.fc1 = nn.Linear(5184, 124)
        self.fc2 = nn.Linear(124, 64)
        
        if False:
          nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
          nninit.constant(self.conv1.bias, 0.1)
          nninit.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
          nninit.constant(self.conv2.bias, 0.1)
          nninit.xavier_uniform(self.conv3.weight, gain=np.sqrt(2))
          nninit.constant(self.conv3.bias, 0.1)
          nninit.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
          nninit.constant(self.fc1.bias, 0.1)
          nninit.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
          nninit.constant(self.fc2.bias, 0.1)

    def forward(self, x):

        #print(x.data.size())

        x = x.view( -1 , 1 , 112, 112)
        x = F.relu( F.max_pool2d( self.conv1(x), 2) )
        x = F.relu( F.max_pool2d( self.conv2(x), 2) )
        x = F.relu( self.conv3(x) )

#        print(x.data.size())

        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)

        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training, p = 0.2)
        x = self.fc2(x)
        #print(x.data.size())
      
        x = x.view( 2, -1 , 64)
        #print(x.data.size())

        return x #F.log_softmax(x)

def obtain(Y_views,labels_views):
    print(Y_views.shape)
    index1 = 0
    mins = []
    for index1 in range( len(Y_views)):
      all_ = []
      for index2 in range( len(Y_views) ):
        if index1 != index2:
          similarity = eval_pair(Y_views[index1], None, Y_views[index2], None,[0,0,0])
          all_.append(similarity)
          #print(str(index1) + " to " + str(index2) + " -> " +  str(similarity))
      mins.append(np.min(all_))

    print(mins)
    print(np.max(mins))
    print(np.min(mins))


if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  print("Loading data")
  #_,_,_,_,_,Y_multiview,labels_views = load_dataset()
  _,Y_multiview,labels_views = load_dataset()

  obtain(Y_multiview,labels_views)
  

