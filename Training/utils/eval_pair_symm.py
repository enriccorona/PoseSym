import numpy
import numpy as np
from rotm2quat import quaternion_from_matrix, euler_matrix, quaternion_from_euler
import math
import torch.nn.functional as F
import torch

equivals = np.load('pairs.npy') # 80 views
x,x_2,x_4,y,y_2,y_4,z,z_2,z_4 = equivals

#equivals = np.load('pairs_168.npy') # 168 views
#x,x_2,x_3,x_4,y,y_2,y_3,y_4,z,z_2,z_3,z_4 = equivals

print("Equivalences are")
print(len(x))
print(len(x_2))
print(len(x_4))
print(len(y))
print(len(y_2))
print(len(y_4))
print(len(z))
print(len(z_2))
print(len(z_4))

def eval_pair2(Y_scene, labels_scene, Y_multiview, labels_views, symm = 0 ):
  if labels_scene[0] == labels_views[0]:

    v = quaternion_from_matrix(np.reshape(Y_multiview,[3,3])) 
    s = quaternion_from_matrix(np.reshape(Y_scene,[3,3]))

    v = v/np.sum(np.square(v))
    s = s/np.sum(np.square(s))

    dist2 = 0.5*math.acos( min(1.0, np.square(np.dot(v,s))*2 - 1 ) )
    
    return dist2

  else:
    return -1

def label_camera_probs( probs, list_equivalences):
  for [i,j] in list_equivalences:
    probs[i-1,j-1] = 1
    probs[j-1,i-1] = 1
  return probs

# Symmetry model is defined as vector of length 3: [ order_x, order_y, order_z] where order for each axis can be 0 if there is none, an integer, or np.inf
def Find_dists_given_symmetry( dists, symmetry_model ):
  probs = np.eye(80)
  if np.max(symmetry_model) == 0:
    return dists
  
  order_x,order_y,order_z = symmetry_model

  if order_x == np.Inf:
    probs = label_camera_probs( probs, x)
  elif order_x == 4:
    probs = label_camera_probs( probs, x_4)
  elif order_x == 2:
    probs = label_camera_probs( probs, x_2)

  if order_y == np.Inf:
    probs = label_camera_probs( probs, y)
  elif order_y == 4:
    probs = label_camera_probs( probs, y_4)
  elif order_y == 2:
    probs = label_camera_probs( probs, y_2)

  if order_z == np.Inf:
    probs = label_camera_probs( probs, z)
  elif order_z == 4:
    probs = label_camera_probs( probs, z_4)
  elif order_z == 2:
    probs = label_camera_probs( probs, z_2)

#REFINE:
  for i in range(80):
    ones = (probs[i,:] == 1)
    for j in range(80):
      if probs[i,j] == 1: 
        probs[j,ones] = 1
        probs[ones,j] = 1

#  np.save('probs.npy',probs)

  return Find_dists_given_probs(dists, probs) 

def Find_dists_given_probs( dists, probs ):
  new_dists = np.copy(dists)

#  np.save('dists.npy',dists)
  
  for i in range(new_dists.shape[0]):
    new_dists[i] = np.min( dists[i] * (1-probs[i,:]) + probs[i,:]*dists )
#  np.save('new_dists.npy',new_dists)

  return new_dists


#def Find_dists_given_symmetry2( dists, symmetry_model ):
def eval_pair(Y_scene, labels_scene, Y_multiview, labels_views, symmetry_model):

  order_x,order_y,order_z = symmetry_model

  # Compute set of all rotations RxRyRz 
  rotations = []
  #num_points = 100
  num_points = 20
 
  if order_x == np.Inf:
    order_x = num_points
  if order_y == np.Inf:
    order_y = num_points
  if order_z == np.Inf:
    order_z = num_points

  if order_x == 0:
    order_x = 1
  if order_y == 0:
    order_y = 1
  if order_z == 0:
    order_z = 1

  for x in range(int(order_x)):
    for y in range(int(order_y)):
      for z in range(int(order_z)):
        rotations.append(np.dot( euler_matrix( x*math.pi*2/order_x , y*math.pi*2/order_y , z*math.pi*2/order_z )[:3,:3], Y_scene ))

  minim = 10.0

  rotations = np.reshape(rotations,[-1,3,3])
  for rot in rotations:
      q = quaternion_from_matrix(rot)
      v = quaternion_from_matrix(np.reshape(Y_multiview,[3,3])) 

      v = v/np.sum(np.square(v))
      q = q/np.sum(np.square(q))

      dist = 0.5*math.acos( min(1.0, np.square(np.dot(q,v))*2 - 1 ) )

      if dist < minim:
        minim = dist

  return minim

def get_symmetry_probs(predictions, axis, other):
 # probs = np.zeros([80,80])
#  probs2 = np.zeros([80,80]) #

#  probs = label_camera_probs( probs, axis)
#  probs2 = label_camera_probs(probs2 ,other)#

#  probs -= np.logical_and((probs == 0)*1, 1*(probs2 == 1) )*1#

  #print(probs)

#  for i in range(80):
#    ones = (probs[i,:] == 1)
#    for j in range(80):
#      if probs[i,j] == 1: 
#        probs[j,ones] = 1
#        probs[ones,j] = 1
  
#  probs = probs - np.eye(80)*(probs>0) #
#  vals = np.float64(predictions*(probs==1)) + np.float64((1-predictions)*(probs==-1)) #
#  prob = vals.sum()/(probs!=0).sum() #
  
  # THIS WORKS:
 # probs = probs - np.eye(80)*(probs > 0)
 # vals = np.float64(predictions*probs)
 # prob = (predictions*probs).sum()/probs.sum()

  #from IPython import embed

  #print(prob)
# TODO: HERE!
  prob = predictions[np.array(axis)[:,0]-1, np.array(axis)[:,1] - 1].mean()
  #embed()
 # print(prob)
  #prob = np.mean(predictions[np.array(axis)-1])

#  prob = np.prod( vals[vals!=0] )
#  print(prob)
#  if prob > max( 0.5**probs.sum(), 1e-80):
#    prob = 1
#  print(probs.sum())
#  print(prob)

  return prob


# Take into account that with only 80 views from dodecahedron we can only infer 2-order, 4-order or infinite-order symmetries.
def infer_symmetry(predictions,tonorm = True): # We are given dot product predictions between views as an 80x80 matrix

  mu = 0.998 #0.9999 #np.mean(predictions) #0.01
  sigma = 0.001 #0.00002 #np.std(predictions)

  prob = 0.01

  if tonorm:
    predictions = (predictions - mu)/sigma  

    # Sigmoid to convert dot products into probabilities
    predictions = 1/( 1 + np.exp(-predictions) )

  equivals = np.load('pairs.npy')
  x,x_2,x_4,y,y_2,y_4,z,z_2,z_4 = equivals

  #x = [k for k in x if k not in x_2 ]
  #y = [k for k in y if k not in y_2 ]
  #z = [k for k in z if k not in z_2 ]

  debug = False
  #debug = True #False

  if debug:
    print("Equivalences are")
    print(len(x))
    print(len(x_2))
    print(len(x_4))
    print(len(y))
    print(len(y_2))
    print(len(y_4))
    print(len(z))
    print(len(z_2))
    print(len(z_4))


  # Conditional probs to get symmetry
  Px2 = get_symmetry_probs(predictions, x_2, x)
  Px4 = get_symmetry_probs(predictions, x_4, x)
  Pxinf = get_symmetry_probs(predictions, x, x)

  Py2 = get_symmetry_probs(predictions, y_2, y)
  Py4 = get_symmetry_probs(predictions, y_4, y)
  Pyinf = get_symmetry_probs(predictions, y, y)

  Pz2 = get_symmetry_probs(predictions, z_2,z)
  Pz4 = get_symmetry_probs(predictions, z_4,z)
  Pzinf = get_symmetry_probs(predictions, z,z)

# Computing lower bounds:
#  Px2 = Px2 - np.sqrt( np.log(prob) / -2/len(x_2)/np.log(2) )
#  Px4 = Px4 - np.sqrt( np.log(prob) / -2/len(x_4)/np.log(2) )
#  Pxinf = Pxinf - np.sqrt( np.log(prob) / -2/len(x)/np.log(2) )
#
#  Py2 = Py2 - np.sqrt( np.log(prob) / -2/len(y_2)/np.log(2) )
#  Py4 = Py4 - np.sqrt( np.log(prob) / -2/len(y_4)/np.log(2) )
#  Pyinf = Pyinf - np.sqrt( np.log(prob) / -2/len(y)/np.log(2) )
#
#  Pz2 = Pz2 - np.sqrt( np.log(prob) / -2/len(z_2)/np.log(2) )
#  Pz4 = Pz4 - np.sqrt( np.log(prob) / -2/len(z_4)/np.log(2) )
#  Pzinf = Pzinf - np.sqrt( np.log(prob) / -2/len(z)/np.log(2) )


  if debug:
    print(Px2)
    print(Px4)
    print(Pxinf)

    print(Py2)
    print(Py4)
    print(Pyinf)

    print(Pz2)
    print(Pz4)
    print(Pzinf)


  symmetry_ = 1*np.array([[Px2>0.5,Px4>0.5,Pxinf>0.5],[Py2>0.5,Py4>0.5,Pyinf>0.5],[Pz2>0.5,Pz4>0.5,Pzinf>0.5]])
  #if debug:
    #print(symmetry_)

  #symmetry_ = 1*np.array([[Px2>0.5, (Pxinf>0.5 and Pxinf > Pyinf and Pxinf > Pzinf) or (Px4>0.5 and Pyinf < 0.5 and Pzinf < 0.5),Pxinf>0.5 and Pxinf > Pyinf and Pxinf > Pzinf],[Py2>0.5,(Py4 > 0.5 and Pyinf > Pxinf and Pyinf > Pzinf) or (Py4>0.5 and Pxinf < 0.5 and Pzinf < 0.5),Pyinf>0.5 and Pyinf > Pxinf and Pyinf > Pzinf],[Pz2>0.5, (Pzinf > 0.5 and Pzinf > Pxinf and Pzinf > Pyinf) or (Pz4>0.5 and Pxinf < 0.5 and Pyinf < 0.5),(Pzinf>0.5 and Pxinf < Pzinf and Pyinf < Pzinf)]])

  if debug:
    print(symmetry_)

  return symmetry_

def infer_symmetry_energy(order_x,order_y,order_z,loss = False):

# HOW TO FIND THIS:
# list_impossible = [[3,3,3],[3,3,0],[3,3,1],[3,3,2],  [3,2,1],[3,2,0],[3,1,0], [2,2,1],[2,2,0],[2,2,3], [2,1,0] ]
  #combinations = []
  #   for i in range(4):
  #       for j in range(4):
  #           for k in range(4):
  #               if [i,j,k] not in list_impossible and [i,k,j] not in list_impossible and [j,i,k] not in list_impossible and [j,k,i] not in list_impossible and [k,i,j] not inlist_impossible and [k,j,i] not in list_impossible:
  #                 combinations.append([i,j,k])

# With 80:
  combinations = [[0, 0, 0],
 [0, 0, 1],
 [0, 0, 2],
 [0, 0, 3],
 [0, 1, 0],
 [0, 1, 1],
 [0, 2, 0],
 [0, 3, 0],
 [1, 0, 0],
 [1, 0, 1],
 [1, 1, 0],
 [1, 1, 1],
 [1, 1, 2],
 [1, 1, 3],
 [1, 2, 1],
 [1, 3, 1],
 [2, 0, 0],
 [2, 1, 1],
 [2, 2, 2],
 [3, 0, 0],
 [3, 1, 1]]
  
# 21 out of 65 possibilities

  energies = []
  for i,j,k in combinations:
    energies.append(order_x[i] + order_y[j] + order_z[k])
 
  #print(energies)
  #print(combinations[np.argmax(energies)])
  if not loss:
    return combinations[np.argmax(energies)] 

  energies = torch.stack(energies)
  aprime = F.softmax(energies)
  #print(combinations[np.argmax(energies.cpu().data.numpy())])
  return combinations[np.argmax(energies)], aprime[np.argmax(energies)]
#  loss = 

def norm_sigmoid(prediction):
  mu = 0.998 #0.9999 #np.mean(predictions) #0.01
  sigma = 0.001 #0.00002 #np.std(predictions)
  prediction = (prediction - mu)/sigma  
  prediction = 1/( 1 + np.exp(-prediction) )
  return prediction

def find_equivalences_from_symm(symmetry):
  probs = np.eye(80)
#  probs2 = np.zeros([80,80]) #
  order_x,order_y,order_z = symmetry

  #equivals = np.load('pairs_168.npy') # 168 views
  #x,x_2,x_3,x_4,y,y_2,y_3,y_4,z,z_2,z_3,z_4 = equivals


  if order_x == np.Inf:
    probs = label_camera_probs( probs, x)
  elif order_x == 4:
    probs = label_camera_probs( probs, x_4)
  elif order_x == 2:
    probs = label_camera_probs( probs, x_2)

  if order_y == np.Inf:
    probs = label_camera_probs( probs, y)
  elif order_y == 4:
    probs = label_camera_probs( probs, y_4)
  elif order_y == 2:
    probs = label_camera_probs( probs, y_2)

  if order_z == np.Inf:
    probs = label_camera_probs( probs, z)
  elif order_z == 4:
    probs = label_camera_probs( probs, z_4)
  elif order_z == 2:
    probs = label_camera_probs( probs, z_2)

#REFINE:
  for i in range(80):
    ones = (probs[i,:] == 1)
    for j in range(80):
      if probs[i,j] == 1: 
        probs[j,ones] = 1
        probs[ones,j] = 1

  return probs

def get_relevant_pairs_80v():
    equivals = np.load('pairs.npy') # 80 views
    x,x_2,x_4,y,y_2,y_4,z,z_2,z_4 = equivals

    relevant_pairs = x[:]
    for i in range(len(y)):
      if y[i] not in relevant_pairs:
        relevant_pairs.append(y[i])
    for i in range(len(z)):
      if z[i] not in relevant_pairs:
        relevant_pairs.append(z[i])

    return relevant_pairs

