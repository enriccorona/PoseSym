import numpy as np
from scipy.misc import imrotate

def random_transforms2(img1,img2,var):
#  return img1,img2
  x = np.random.randint(0,100)
  y = np.random.randint(0,100)

  size_x = np.random.randint(5,10)
  size_y = np.random.randint(5,10)

  img1[x:x+size_x, y:y+size_y,0] = 0

  return img1,img2

  if var == 0:
    return img1,img2
  elif var == 1:
    return imrotate(img1,90),imrotate(img2,90)
  elif var == 2:
    return imrotate(img1,180),imrotate(img2,180)
  else:
    return imrotate(img1,270),imrotate(img2,270)
    
def random_transforms(img1,img2,var):
  return img1,img2
  img1 = np.reshape(img1,[112,112])
  img2 = np.reshape(img2,[112,112])
  if var == 0:
    return img1.reshape([112,112,1]),img2.reshape([112,112,1])
  elif var == 1:
    return imrotate(img1,90).reshape([112,112,1]).reshape([112,112,1]),imrotate(img2,90).reshape([112,112,1])
  elif var == 2:
    return imrotate(img1,180).reshape([112,112,1]),imrotate(img2,180).reshape([112,112,1])
  else:
    return imrotate(img1,270).reshape([112,112,1]),imrotate(img2,270).reshape([112,112,1])
