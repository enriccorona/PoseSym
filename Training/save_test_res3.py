# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines

def save_results(image, batch, pred_order, order_sim, index1, sim = None, bestview = None):
      
      plt.close()
      plt.imshow( (np.float32(image)/255)  ) #batch[pred_order.cpu().data.numpy()[0,0],0].reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_0_'+str(index1)+'.png') #plt.pause(3)

      plt.close()
      plt.imshow( bestview.reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_1_'+str(index1)+'.png') #plt.pause(3)

      plt.close()
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,0]].reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_2_'+str(index1)+'.png') #plt.pause(3)

      plt.close()
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,1]].reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_3_'+str(index1)+'.png') #plt.pause(3)

      plt.close()
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,2]].reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_4_'+str(index1)+'.png') #plt.pause(3)

      plt.close()
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,3]].reshape([112,112]) )
      plt.axis('off')
      plt.savefig('test_res3/image_5_'+str(index1)+'.png') #plt.pause(3)

  #      plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)


