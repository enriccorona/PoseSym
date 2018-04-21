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
from eval_pair import eval_pair
from transforms import random_transforms
from get_data_depth_conserving import load_dataset
from get_data_more_views import load_views

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
      
        return x #F.log_softmax(x)

def train(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model.train()
    tloss = 0
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0
    iterations = 1000

    for index1 in range( iterations): 
      ind = np.arange(len(X))
      np.random.shuffle(ind)
      X = X[ind]
      Y = Y[ind]
      labels = labels[ind]

      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]

      #if len(imgs) == 0:
      #  continue
      batch = []
      sim = []
      var = np.random.randint(4)
      for index2 in range( len(imgs) ):
        num_cam = lab[index2,1]
        similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2])

        img1 = X[index1].reshape([img_size,img_size,1])
        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(random_transforms(img1,img2,var))
        sim.append(similarity)

      batch = np.array(batch)
      sim = np.array(sim)
      sim = 0.5*sim/(np.max(sim) - np.min(sim))

      input_net = np.float32( [ np.transpose(batch[:,0],[0,3,1,2]), np.transpose(batch[:,1],[0,3,1,2]) ] )
        
      data = torch.from_numpy( input_net )
      target = torch.from_numpy( np.float32(sim) )

      if args.cuda:
          data, target = data.cuda(), target.cuda()

      data, target = Variable(data), Variable(target)

      optimizer.zero_grad()
      output = model(data)

      output = torch.mul(output[0,:,:],output[1,:,:])
      output = torch.sum(output, 1)
      
      output = output.view( 1 , -1)
      
      output = F.softmax(output)
      
      _, pred_order = torch.sort(-1*output,1)

      order_sim = np.argsort(sim,0)
      #output = output.view( 1 , -1)
      
      #loss = F.nll_loss(output, Variable( torch.from_numpy( np.float32( [order_sim[0]] ) ).cuda().long() ) )
      loss = torch.max( Variable( torch.zeros(len(sim)).cuda() ) , alpha + output.view(-1) - output[0,order_sim[0]].repeat(80) )
      loss = torch.mul( Variable( torch.FloatTensor(sim - sim[order_sim[0]]).cuda() ), loss)
      loss = loss.sum()

      # Compute gradients and train
      loss.backward()
      optimizer.step()

      acc += 1*(pred_order.cpu().data.numpy()[0,0] == order_sim[0])
      r3 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:3])
      r5 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:5])
      r10 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:10])

      #if acc > 0:
      #  print([pred_order.cpu().data.numpy()[0,0],order_sim[0]])
      #  print(acc)
      
      tloss += loss.cpu().data.numpy()[0]
      
    print('Train Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, iterations, tloss/iterations, acc*1.0/iterations, r3*1.0/iterations, r5*1.0/iterations, r10*1.0/iterations ))

def test(epoch,X,Y,labels,X_views,Y_views,labels_views):
    model.eval()
    
    tloss = 0
    correct = 0
    
    test_batch = 1
    acc = 0
    r3 = 0
    r5 = 0
    r10 = 0

    for index1 in range( len(X) ):
      
      ind = np.arange(len(X))
      np.random.shuffle(ind)

      X = X[ind]
      Y = Y[ind]
      labels = labels[ind]

      mesh = labels[index1,0]
      cam = labels[index1,2]

      # RETRIEVE IN DATABASE RAW CONTAINING MESH
      nums_mesh = (labels_views[:,0] == mesh)
      imgs = X_views[nums_mesh]
      lab = labels_views[nums_mesh]

      batch = []
      sim = []
      var = np.random.randint(4)
      for index2 in range( len(imgs) ):
        num_cam = lab[index2,1]
        similarity = eval_pair(Y[index1], labels[index1], Y_views[num_cam-1], lab[index2])

        img1 = X[index1].reshape([img_size,img_size,1])
        img2 = imgs[index2].reshape([img_size,img_size,1])
        batch.append(random_transforms(img1,img2,var))
        sim.append(similarity)

      batch = np.array(batch)
      sim = np.array(sim)
      sim = 0.5*sim/(np.max(sim) - np.min(sim))

      input_net = np.float32( [ np.transpose(batch[:,0],[0,3,1,2]), np.transpose(batch[:,1],[0,3,1,2]) ] )

      data = torch.from_numpy( input_net )
      target = torch.from_numpy( np.float32(sim) )
      
      if args.cuda:
          data, target = data.cuda(), target.cuda()
      
      data, target = Variable(data), Variable(target)
      
      optimizer.zero_grad()
      output = model(data)

      output = torch.mul(output[0,:,:],output[1,:,:])
      
      output = torch.sum(output, 1)
      output = output.view( 1 , -1)
      output = F.softmax(output)
      
      _, pred_order = torch.sort(-1*output,1)
      
      order_sim = np.argsort(sim,0)
      
      loss = torch.max( Variable( torch.zeros(len(sim)).cuda() ) , alpha + output.view(-1) - output[0,order_sim[0]].repeat(80) )
      loss = torch.mul( Variable( torch.FloatTensor(sim - sim[order_sim[0]]).cuda() ), loss)
      loss = loss.sum()


      acc += 1*(pred_order.cpu().data.numpy()[0,0] == order_sim[0])
      r3 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:3])
      r5 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:5])
      r10 += 1*(order_sim[0] in pred_order.cpu().data.numpy()[0,0:10])

      tloss += loss.cpu().data.numpy()[0]

    print('Test Epoch {} containing {:.0f} iterations \tLoss: {:.6f} - Acc: {:.3f}, R@3: {:.3f}, R@5: {:.3f}, R@10: {:.3f}'.format(
              epoch, len(X), tloss/len(X), acc*1.0/len(X), r3*1.0/len(X), r5*1.0/len(X), r10*1.0/len(X) ))
    return acc*1.0/len(X)
    
if __name__ == "__main__":
  epochs = 5000
  alpha = 0.1

  print("Loading data")
  X_views,Y_multiview,labels_views = load_views()

  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  args = parser.parse_args()

  torch.manual_seed(args.seed)

  args.cuda = True

  kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

  model = torch.load('model_train_2') #Encoder()

  meshes = np.unique(labels_views[:,0])
  print(meshes)

  results = []  
  meshs = []
  for mesh in meshes:
    coinc = (labels_views[:,0] == mesh)
    X = X_views[coinc]
    cams = labels_views[coinc,1]
    
    if( len(cams) < 80 ):
      continue
    
    vectors = []
    rot = []
    cams_ = []
    for ind in range(80):
      cam = cams[ind]

      data = torch.from_numpy(np.float32(X[ind].reshape([1,1,1,img_size,img_size]) ) )
      data = Variable(data.cuda())

      output = model(data)
      vectors.append( output.cpu().data.numpy().reshape(-1) )
      cams_.append(cam)

#      rot.append(Y_multiview[cam-1])
    
    vectors = np.array(vectors)
    rot = np.array(rot)
    cams_ = np.array(cams_)
    print(cams_)
    ind = np.argsort(cams_)

    cams_ = cams_[ind]
    vectors = vectors[ind]
    rot = vectors[ind]
    print(cams_)

    arr = []
    for ind in range(80):
      row = []
#      for ind2 in range(ind + 1,80):
#      for ind2 in range(ind,80):
      for ind2 in range(80):
        pred = np.sum( vectors[ind]*vectors[ind2] )
#        pred = np.sum( np.square(vectors[ind] - vectors[ind2] ))
        row.append(pred)
      arr.append(np.array(row))

    results.append(np.array(arr))
    meshs.append(mesh)

  results = np.array(results)
  meshs = np.array(meshs)

  np.save('meshs_fc.npy',meshs)
  np.save('results_fc.npy',results)

