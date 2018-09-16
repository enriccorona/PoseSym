from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from utils.rotm2quat import quaternion_from_matrix
from utils.eval_pair import eval_pair
from utils.transforms import random_transforms

class Encoder_S(nn.Module):
    def __init__(self):
        import nninit
        super(Encoder_S, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride = 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride = 1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride = 1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride = 1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=2, stride = 1)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 124)
        #self.fc1 = nn.Linear(128, 124)
        #self.fc1 = nn.Linear(5184, 124)
        self.fc2 = nn.Linear(124, 64)

        if True: #False:
          nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
          nninit.constant(self.conv1.bias, 0.1)
          nninit.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
          nninit.constant(self.conv2.bias, 0.1)
          nninit.xavier_uniform(self.conv3.weight, gain=np.sqrt(2))
          nninit.constant(self.conv3.bias, 0.1)
          nninit.xavier_uniform(self.conv4.weight, gain=np.sqrt(2))
          nninit.constant(self.conv4.bias, 0.1)
          nninit.xavier_uniform(self.conv5.weight, gain=np.sqrt(2))
          nninit.constant(self.conv5.bias, 0.1)
          nninit.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
          nninit.constant(self.fc1.bias, 0.1)
          nninit.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
          nninit.constant(self.fc2.bias, 0.1)

    def forward(self, x):

        x = x.view( -1 , 3 , 112, 112)
        x = F.relu( F.max_pool2d( self.conv1(x), 2) )
        x = F.relu( F.max_pool2d( self.conv2(x), 2) )
        x = F.relu( self.conv3(x) )
        x = F.relu( self.conv4(x) )
        x = F.relu( self.conv5(x) )

        #print(x.data.size())

        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

#NORMALIZE

        #print(x.data.size())
        k = torch.sum( x*x, 1 )
        x = x/(torch.sqrt(k.view([-1,1])).repeat(1,64) ) 
 
        return x

class Encoder_V(nn.Module):
    def __init__(self):
        import nninit
        super(Encoder_V, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride = 1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride = 1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride = 1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=2, stride = 1)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 124)
        #self.fc1 = nn.Linear(128, 124)
        #self.fc1 = nn.Linear(5184, 124)
        self.fc2 = nn.Linear(124, 64)
        
        if True: #False:
          nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
          nninit.constant(self.conv1.bias, 0.1)
          nninit.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
          nninit.constant(self.conv2.bias, 0.1)
          nninit.xavier_uniform(self.conv3.weight, gain=np.sqrt(2))
          nninit.constant(self.conv3.bias, 0.1)
          nninit.xavier_uniform(self.conv4.weight, gain=np.sqrt(2))
          nninit.constant(self.conv4.bias, 0.1)
          nninit.xavier_uniform(self.conv5.weight, gain=np.sqrt(2))
          nninit.constant(self.conv5.bias, 0.1)
          nninit.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
          nninit.constant(self.fc1.bias, 0.1)
          nninit.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
          nninit.constant(self.fc2.bias, 0.1)

    def forward(self, x):

        x = x.view( -1 , 1 , 112, 112)
        x = F.relu( F.max_pool2d( self.conv1(x), 2) )
        x = F.relu( F.max_pool2d( self.conv2(x), 2) )
        x = F.relu( self.conv3(x) )
        x = F.relu( self.conv4(x) )
        x = F.relu( self.conv5(x) )

       # print(x.data.size())

        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

#NORMALIZE

        k = torch.sum( x*x, 1 )
        x = x/(torch.sqrt(k.view([-1,1])).repeat(1,64) ) 
        #k = torch.sum( x*x, 1 )
        #print(k)
        #print(x)

        return x #F.log_softmax(x)
