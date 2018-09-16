import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_results(image, batch, pred_order, order_sim, index1):
      plt.close()
      draws = 4 #8

      fig = plt.figure(figsize=(10, 4))
      plt.subplot(1,draws,1)

      if len(np.shape(image)) == 3:
        if np.shape(image)[2] == 3:
          plt.imshow( (np.float32(image[:,:,::-1])/255)  ) #batch[pred_order.cpu().data.numpy()[0,0],0].reshape([112,112]) )
        else:
          plt.imshow( image.reshape([112,112]) ) #batch[pred_order.cpu().data.numpy()[0,0],0].reshape([112,112]) )

      else:
        plt.imshow( image.reshape([112,112]) ) #batch[pred_order.cpu().data.numpy()[0,0],0].reshape([112,112]) )

      plt.axis('off')
      plt.subplots_adjust(hspace = 0.2)
      
      plt.subplot(1,draws,2)
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,0]].reshape([112,112]) )
      plt.axis('off')
      
      if ( pred_order.cpu().data.numpy()[0,0] in order_sim ):
        c = 'g'
      else:
        c = 'r'

      plt.figtext(0.335,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)
      
      plt.subplot(1,draws,3)
      
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,1]].reshape([112,112]) ) 
      plt.axis('off')
      
      if ( pred_order.cpu().data.numpy()[0,1] in order_sim ):
        c = 'g'
      else:
        c = 'r'
      
      plt.figtext(0.535,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

      plt.subplot(1,draws,4)
      plt.imshow( batch[pred_order.cpu().data.numpy()[0,2]].reshape([112,112]) ) 
      plt.axis('off')

      if ( pred_order.cpu().data.numpy()[0,2] in order_sim ):
        c = 'g'
      else:
        c = 'r'

      plt.figtext(0.738,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

      if False:

        plt.subplot(1,draws,5)
        plt.imshow( batch[pred_order.cpu().data.numpy()[0,3]].reshape([112,112]) ) 
        plt.axis('off')

        if ( pred_order.cpu().data.numpy()[0,3] in order_sim ):
          c = 'g'
        else:
          c = 'r'

  #      plt.figtext(0.738,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

        #plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)


        plt.subplot(1,draws,6)
        plt.imshow( batch[pred_order.cpu().data.numpy()[0,4]].reshape([112,112]) ) 
        plt.axis('off')

        if ( pred_order.cpu().data.numpy()[0,4] in order_sim ):
          c = 'g'
        else:
          c = 'r'

        #plt.figtext(0.738,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

        #plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)

        plt.subplot(1,draws,7)
        plt.imshow( batch[pred_order.cpu().data.numpy()[0,5]].reshape([112,112]) ) 
        plt.axis('off')

        if ( pred_order.cpu().data.numpy()[0,6] in order_sim ):
          c = 'g'
        else:
          c = 'r'

       # plt.figtext(0.738,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

       # plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)

        plt.subplot(1,draws,8)
        plt.imshow( batch[pred_order.cpu().data.numpy()[0,6]].reshape([112,112]) ) 
        plt.axis('off')

        if ( pred_order.cpu().data.numpy()[0,6] in order_sim ):
          c = 'g'
        else:
          c = 'r'

        #plt.figtext(0.738,0.20,'aaaaaaaaaaaaaaaaaa',color = c,backgroundcolor=c)

      plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)












  #      plt.savefig('test_res/image'+str(index1)+'.png') #plt.pause(3)


